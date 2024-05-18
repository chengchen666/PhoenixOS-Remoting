use super::ChannelBufferManager;

use log::info;
use std::net::SocketAddr;
use std::sync::{Mutex, Arc};
use std::io::Result;
use std::thread::JoinHandle;

use KRdmaKit::services_user::{ConnectionManagerServer, DefaultConnectionManagerHandler, MRInfo};
use KRdmaKit::context::{Context};
use KRdmaKit::{MemoryRegion, QueuePairBuilder, QueuePairStatus, UDriver, QueuePair};

struct Server {
    cm: Arc<ConnectionManagerServer<DefaultConnectionManagerHandler>>,
    listener: Arc<Mutex<JoinHandle<std::io::Result<()>>>>,
}

struct Client {
    mr: Arc<MemoryRegion>,
    qp: Arc<QueuePair>,
    rinfo: MRInfo,
}

// connnection type
enum Conn {
    Server(Server),
    Client(Client),
}

pub struct RDMAChannelBufferManager {
    _name: String,
    ptr: *mut u8,
    buf_len: usize,
    conn: Conn,
}

unsafe impl Send for RDMAChannelBufferManager {}

impl RDMAChannelBufferManager {
    pub fn new_server(name: &str, buf_len: usize, addr: SocketAddr) -> Result<Self> {
        let (ctx, mr, ptr) = Self::allocate_mr(buf_len);
        let mut handler = DefaultConnectionManagerHandler::new(&ctx, 1);
        handler.register_mr(vec![(name.to_string(), mr)]);
        let cm = ConnectionManagerServer::new(handler);
        let listener = cm.spawn_listener(addr);

        Ok(Self {
            _name: name.to_string(),
            ptr,
            buf_len,
            conn: Conn::Server(Server {
                cm: cm,
                listener: Arc::new(Mutex::new(listener))
            })
        })
    }

    pub fn new_client(name: &str, buf_len: usize, addr: SocketAddr, client_port: u8) -> Result<Self> {
        let (ctx, mr, ptr) = Self::allocate_mr(buf_len);
        let mut builder = QueuePairBuilder::new(&ctx);
        builder
            .allow_remote_rw()
            .allow_remote_atomic()
            .set_port_num(client_port);
        let qp = builder.build_rc().expect("failed to create the client QP");
        let qp = qp.handshake(addr).expect("Handshake failed!");
        match qp.status().expect("Query status failed!") {
            QueuePairStatus::ReadyToSend => info!("QP bring up succeeded"),
            _ => eprintln!("Error : Bring up failed"),
        }

        let mr_infos = qp.query_mr_info().expect("Failed to query MR info");
        let rinfo = *(mr_infos.inner().get(name).expect("Unregistered MR"));

        Ok(Self {
            _name: name.to_string(),
            ptr,
            buf_len,
            conn: Conn::Client(Client {
                mr: mr.into(),
                qp,
                rinfo
            })
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

impl Drop for RDMAChannelBufferManager {
    fn drop(&mut self) {
        // Just need to handle server side listener specifically.
        // The MemoryRegion with Arc wrapped with be drop automatically.
        if let Conn::Server(server) = &self.conn {
            server.cm.stop_listening();
            match Arc::try_unwrap(server.listener.clone()) {
                Ok(mutex) => {
                    let join_handle = mutex.into_inner().unwrap();
                    let _ = join_handle.join();
                }
                Err(_) => {}
            };
        }
    }
}

impl ChannelBufferManager for RDMAChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.ptr, self.buf_len)
    }

    fn read_at(&self, offset: usize, dst: *mut u8, count: usize) -> usize {
        if let Conn::Client(client) = &self.conn {
            let l: u64 = offset as u64;
            let r: u64 = l + (count * std::mem::size_of::<u8>()) as u64;
            let _ = client.qp.post_send_read(&client.mr, l..r, true, client.rinfo.addr + l, client.rinfo.rkey, l);
            Self::poll_till_completion(&client.qp);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.add(offset) as _, dst, count);
        }
        count
    }

    fn write_at(&self, offset: usize, src: *const u8, count: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.ptr.add(offset) as _, count);
        }
        if let Conn::Client(client) = &self.conn {
            let l: u64 = offset as u64;
            let r: u64 = l + (count * std::mem::size_of::<u8>()) as u64;
            let _ = client.qp.post_send_write(&client.mr, l..r, true, client.rinfo.addr + l, client.rinfo.rkey, l + self.buf_len as u64);
            Self::poll_till_completion(&client.qp);
        }
        count
    }
}
