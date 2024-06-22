use crate::ringbufferchannel::{
    BufferManager, RingBufferChannel, RingBufferManager, HEAD_OFF, META_AREA, TAIL_OFF,
};
use crate::{CommChannelInner, CommChannelError};

use log::info;
use std::io::Result as IOResult;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::ptr;

use KRdmaKit::context::Context;
use KRdmaKit::services_user::{
    ConnectionManagerServer, DefaultConnectionManagerHandler, MRInfo, MRWrapper,
};
use KRdmaKit::{
    ControlpathError::CreationError, MemoryRegion, QueuePair, QueuePairBuilder, QueuePairStatus,
    UDriver,
};

const BATCH_SIZE: usize = 16;

pub struct RDMAChannel {
    mr_ptr: *mut u8,
    buf_len: usize,
    mr: Arc<MemoryRegion>,
    qp: Arc<QueuePair>,
    rinfo: MRInfo,
    pending_num: Arc<Mutex<usize>>,
    last_tail: Arc<Mutex<usize>>,
    tail_pos: Arc<Mutex<usize>>,
}

unsafe impl Send for RDMAChannel {}
unsafe impl Sync for RDMAChannel {}

impl RDMAChannel {
    pub fn new_server(name: &str, buf_len: usize, addr: SocketAddr) -> IOResult<Self> {
        let (ctx, mr, mr_ptr) = Self::allocate_mr(buf_len);
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
            if let Some(value) = cm
                .handler()
                .exp_get_remote_mrs()
                .lock()
                .unwrap()
                .inner()
                .get(name)
            {
                break value.clone();
            }
        };

        cm.stop_listening();
        let _ = listener.join();

        let mut handler = Arc::try_unwrap(cm)
            .unwrap_or_else(|_| panic!("Failed to unwrap cm"))
            .into_handler();
        let Some(mr) = handler.registered_mr.inner.remove(name) else {
            panic!()
        };

        Ok(Self::new(mr_ptr, buf_len, mr.into(), qp, rinfo))
    }

    pub fn new_client(
        name: &str,
        buf_len: usize,
        addr: SocketAddr,
        client_port: u8,
    ) -> IOResult<Self> {
        let (ctx, mr, mr_ptr) = Self::allocate_mr(buf_len);
        let mut builder = QueuePairBuilder::new(&ctx);
        builder
            .allow_remote_rw()
            .allow_remote_atomic()
            .set_port_num(client_port);
        let qp = loop {
            let qp = builder
                .clone()
                .build_rc()
                .expect("failed to create the client QP");
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

        let Some(mr) = mr_wrapper.inner.remove(name) else {
            panic!()
        };

        Ok(Self::new(mr_ptr, buf_len, mr.into(), qp, rinfo))
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

    #[inline]
    pub fn get_last_tail(&self) -> usize {
        *self.last_tail.lock().unwrap()
    }

    #[inline]
    pub fn set_last_tail(&self, last_tail: usize) {
        *self.last_tail.lock().unwrap() = last_tail;
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
        let mr_ptr = mr.get_virt_addr() as *mut u8;
        (ctx, mr, mr_ptr)
    }

    fn new(
        mr_ptr: *mut u8,
        buf_len: usize,
        mr: Arc<MemoryRegion>,
        qp: Arc<QueuePair>,
        rinfo: MRInfo,
    ) -> Self {
        Self {
            mr_ptr,
            buf_len,
            mr,
            qp,
            rinfo,
            pending_num: Arc::new(Mutex::new(0)),
            last_tail: Arc::new(Mutex::new(0)),
            tail_pos: Arc::new(Mutex::new(1)),
        }
    }
}

impl RDMAChannel {
    fn get_req_id(&self) -> u64 {
        lazy_static! {
            static ref REQ_ID: Mutex<u64> = Mutex::new(0);
        }
        *REQ_ID.lock().unwrap() += 1;
        *REQ_ID.lock().unwrap()
    }

    fn read_remote(&self, offset: usize, len: usize) -> usize {
        for _ in 0..self.get_pending_num() {
            Self::poll_till_completion(&self.qp);
        }
        self.set_pending_num(0);

        let l: u64 = offset as u64;
        let r: u64 = l + len as u64;
        let _ = self.qp.post_send_read(
            &self.mr,
            l..r,
            true,
            self.rinfo.addr + l,
            self.rinfo.rkey,
            self.get_req_id(),
        );
        Self::poll_till_completion(&self.qp);
        len
    }

    fn write_remote(&self, offset: usize, len: usize) -> usize {
        let l: u64 = offset as u64;
        let r: u64 = l + len as u64;
        let _ = self.qp.post_send_write(
            &self.mr,
            l..r,
            true,
            self.rinfo.addr + l,
            self.rinfo.rkey,
            self.get_req_id(),
        );

        self.set_pending_num(self.get_pending_num() + 1);
        if self.get_pending_num() == BATCH_SIZE {
            Self::poll_batch(&self.qp);
            self.set_pending_num(0);
        }
        len
    }

    fn write_tail_remote(&self, tail: usize) {
        let len = std::mem::size_of::<usize>();
        let t: u64 = TAIL_OFF as u64;
        let l = t + (*self.tail_pos.lock().unwrap() as u64 * len as u64);
        let r: u64 = l + len as u64;
        unsafe { ptr::write_volatile(self.get_ptr().add(l as usize) as *mut usize, tail) }

        let _ = self.qp.post_send_write(
            &self.mr,
            l..r,
            true,
            self.rinfo.addr + t,
            self.rinfo.rkey,
            self.get_req_id(),
        );
        *self.tail_pos.lock().unwrap() += 1;
        if *self.tail_pos.lock().unwrap() == 8 {
            *self.tail_pos.lock().unwrap() = 1;
        }
        self.set_pending_num(self.get_pending_num() + 1);
        if self.get_pending_num() == BATCH_SIZE {
            Self::poll_batch(&self.qp);
            self.set_pending_num(0);
        }
    }
}

impl BufferManager for RDMAChannel {
    fn get_ptr(&self) -> *mut u8 {
        self.mr_ptr
    }

    fn get_len(&self) -> usize {
        self.buf_len
    }
}

impl RingBufferManager for RDMAChannel {}

impl RingBufferChannel for RDMAChannel {}

impl CommChannelInner for RDMAChannel {
    fn flush_out(&self) -> Result<(), CommChannelError> {
        let cur_tail = self.read_tail_volatile();
        let last_tail = self.get_last_tail();
        if last_tail < cur_tail {
            self.write_remote(META_AREA + last_tail, cur_tail - last_tail);
        }
        if cur_tail < last_tail {
            self.write_remote(META_AREA + last_tail, self.capacity() - last_tail);
            self.write_remote(META_AREA, cur_tail);
        }

        self.set_last_tail(cur_tail);
        self.write_tail_volatile(cur_tail);
        self.write_tail_remote(cur_tail);

        while self.is_full() {
            self.read_remote(HEAD_OFF, std::mem::size_of::<usize>());
        }
        Ok(())
    }
}
