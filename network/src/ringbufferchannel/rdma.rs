use super::ChannelBufferManager;

use log::info;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::io::Result;

use KRdmaKit::services_user::{ConnectionManagerServer, DefaultConnectionManagerHandler, MRInfo, MRWrapper};
use KRdmaKit::context::{Context};
use KRdmaKit::{MemoryRegion, QueuePairBuilder, QueuePairStatus, UDriver, QueuePair, ControlpathError::CreationError};

const BATCH_SIZE: usize = 16;

pub struct RDMAChannelBufferManager {
    _name: String,
    ptr: *mut u8,
    buf_len: usize,
    mr: Arc<MemoryRegion>,
    qp: Arc<QueuePair>,
    rinfo: MRInfo,
    pending_num: Arc<Mutex<usize>>,
}

unsafe impl Send for RDMAChannelBufferManager {}

impl RDMAChannelBufferManager {
    pub fn new_server(name: &str, buf_len: usize, addr: SocketAddr) -> Result<Self> {
        let (ctx, mr, ptr) = Self::allocate_mr(buf_len);
        let mut handler = DefaultConnectionManagerHandler::new(&ctx, 1);
        handler.register_mr(vec![(name.to_string(), mr)]);
        let cm = ConnectionManagerServer::new(handler);
        let listener = cm.spawn_listener(addr);

        // Wait client side connection, then get qp.
        let qp = loop {
            if let Some(qp) = cm.handler().exp_get_qps().get(0) {
                break qp.clone();
            }
        };
        // Wait to get client side mr info.
        let rinfo = loop {
            if let Some(value) = cm.handler().exp_get_remote_mrs().lock().unwrap().inner().get(name) {
                break value.clone();
            }
        };

        cm.stop_listening();
        let _ = listener.join();

        let mut handler = Arc::try_unwrap(cm).unwrap_or_else(|_| panic!("Failed to unwrap cm")).into_handler();
        let Some(mr) = handler.registered_mr.inner.remove(name) else { panic!() };

        Ok(Self {
            _name: name.to_string(),
            ptr,
            buf_len,
            mr: mr.into(),
            qp,
            rinfo,
            pending_num: Arc::new(Mutex::new(0)),
        })
    }

    pub fn new_client(name: &str, buf_len: usize, addr: SocketAddr, client_port: u8) -> Result<Self> {
        let (ctx, mr, ptr) = Self::allocate_mr(buf_len);
        let mut builder = QueuePairBuilder::new(&ctx);
        builder
            .allow_remote_rw()
            .allow_remote_atomic()
            .set_port_num(client_port);
        let qp = loop {
            let qp = builder.clone().build_rc().expect("failed to create the client QP");
            match qp.handshake(addr) {
                Ok(res) => {
                    break res;
                }
                Err(e) => {
                    if let CreationError(msg, _) = &e {
                        if *msg == "Failed to connect server" {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                            continue;
                        }
                    }
                    panic!("Handshake failed!");
                }
            }
        };
        match qp.status().expect("Query status failed!") {
            QueuePairStatus::ReadyToSend => info!("QP bring up succeeded"),
            _ => eprintln!("Error : Bring up failed"),
        }

        let mr_infos = qp.query_mr_info().expect("Failed to query MR info");
        let rinfo = *(mr_infos.inner().get(name).expect("Unregistered MR"));

        // Send client side mr info to server.
        let mrs = vec![(name.to_string(), mr)];
        let mut mr_wrapper: MRWrapper = Default::default();
        mr_wrapper.insert(mrs);
        let mr_info = mr_wrapper.to_mrinfos();
        let _ = qp.send_mr_info(mr_info).unwrap();

        let Some(mr) = mr_wrapper.inner.remove(name) else { panic!() };

        Ok(Self {
            _name: name.to_string(),
            ptr,
            buf_len,
            mr: mr.into(),
            qp,
            rinfo,
            pending_num: Arc::new(Mutex::new(0)),
        })
    }

    /// A simple loop queue pair poll to poll completion queue synchronously.
    pub fn poll_till_completion(qp: &Arc<QueuePair>) -> i32 {
        let mut completions = [Default::default()];
        loop {
            if let Ok(ret) = qp.poll_send_cq(&mut completions) {
                if ret.len() > 0 {
                    break;
                }
            } else {
                return -1;
            }
        }
        return 0;
    }

    pub fn poll_batch(qp: &Arc<QueuePair>) -> i32 {
        let mut completions = [Default::default(); BATCH_SIZE];
        loop {
            if let Ok(ret) = qp.poll_send_cq(&mut completions) {
                if ret.len() > 0 {
                    break;
                }
            } else {
                return -1;
            }
        }
        return 0;
    }

    #[inline]
    pub fn get_pending_num(&self) -> usize {
        *self.pending_num.lock().unwrap()
    }

    #[inline]
    pub fn set_pending_num(&self, pending_num: usize) {
        *self.pending_num.lock().unwrap() = pending_num;
    }

    fn allocate_mr(buf_len: usize) -> (Arc<Context>, MemoryRegion, *mut u8) {
        let ctx = UDriver::create()
            .expect("failed to query device")
            .devices()
            .into_iter()
            .next()
            .expect("no rdma device available")
            .open_context()
            .expect("failed to create RDMA context");
        let mr = MemoryRegion::new(ctx.clone(), buf_len).expect("Failed to allocate MR");
        let ptr = mr.get_virt_addr() as *mut u8;
        (ctx, mr, ptr)
    }
}

impl ChannelBufferManager for RDMAChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.ptr, self.buf_len)
    }

    fn is_remoted(&self) -> bool {
        true
    }

    fn read_at(&self, offset: usize, dst: *mut u8, size: usize, remote: Option<bool>) -> usize {
        let remote = remote.unwrap_or(false);
        if remote {
            let l: u64 = offset as u64;
            let r: u64 = l + (size * std::mem::size_of::<u8>()) as u64;
            let _ = self.qp.post_send_read(&self.mr, l..r, true, self.rinfo.addr + l, self.rinfo.rkey, l);
            Self::poll_till_completion(&self.qp);
        }
        unsafe {
            if dst != self.ptr.add(offset) {
                std::ptr::copy_nonoverlapping(self.ptr.add(offset) as _, dst, size);
            }
        }
        size
    }

    fn write_at(&self, offset: usize, src: *const u8, size: usize, remote: Option<bool>) -> usize {
        let remote = remote.unwrap_or(false);
        unsafe {
            if src != self.ptr.add(offset) {
                std::ptr::copy_nonoverlapping(src, self.ptr.add(offset) as _, size);
            }
        }
        if remote {
            let l: u64 = offset as u64;
            let r: u64 = l + (size * std::mem::size_of::<u8>()) as u64;
            let _ = self.qp.post_send_write(&self.mr, l..r, true, self.rinfo.addr + l, self.rinfo.rkey, l + self.buf_len as u64);
            self.set_pending_num(self.get_pending_num() + 1);
            if self.get_pending_num() == BATCH_SIZE {
                Self::poll_batch(&self.qp);
                self.set_pending_num(0);
            }
        }
        size
    }
}
