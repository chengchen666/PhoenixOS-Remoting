use lazy_static::lazy_static;

use log::{error, info};

#[cfg(feature = "rdma")]
use network::ringbufferchannel::RDMAChannel;

use network::{
    ringbufferchannel::{EmulatorChannel, SHMChannel},
    type_impl::{recv_slice_to, send_slice, MemPtr},
    Channel, CommChannel, Transportable,
};

use codegen::{cuda_hook_hijack, use_thread_local};

pub mod hijack;
pub use hijack::*;

pub mod elf;
use elf::interfaces::{fat_header, kernel_info_t};
use elf::ElfController;

pub mod dl;
pub use dl::*;

use std::cell::RefCell;
use std::io::Read as _;
use std::{
    sync::Mutex,
};

struct ClientThread {
    id: i32,
    channel_sender: Channel,
    channel_receiver: Channel,
    resource_idx: usize,
    #[cfg(feature = "local")]
    cuda_device: Option<std::ffi::c_int>,
}

impl ClientThread {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the client side, the sender's name is ctos_channel_name,
    // receiver's name is stoc_channel_name.
    fn new() -> Self {
        let config = &*network::CONFIG;
        let (id, channel_sender, channel_receiver) = match config.comm_type.as_str() {
            "shm" => {
                let id = {
                    let mut stream = std::net::TcpStream::connect(&config.daemon_socket).unwrap();
                    let mut buf = [0u8; 4];
                    stream.read_exact(&mut buf).unwrap();
                    i32::from_be_bytes(buf)
                };
                log::info!("Client id: {id}");
                let sender =
                    SHMChannel::new_client_with_id(&config.ctos_channel_name, id, config.buf_size)
                        .unwrap();
                let receiver =
                    SHMChannel::new_client_with_id(&config.stoc_channel_name, id, config.buf_size)
                        .unwrap();
                if cfg!(feature = "emulator") {
                    (
                        id,
                        Channel::new(Box::new(EmulatorChannel::new(Box::new(sender)))),
                        Channel::new(Box::new(EmulatorChannel::new(Box::new(receiver)))),
                    )
                } else {
                    (id, Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
                }
            }
            #[cfg(feature = "rdma")]
            "rdma" => {
                // client side sender should connect to server's receiver socket.
                let sender = RDMAChannel::new_client(
                    &config.ctos_channel_name,
                    config.buf_size,
                    config.receiver_socket.parse().unwrap(),
                    1,
                )
                .unwrap();
                // client side receiver should connect to server's sender socket.
                let receiver = RDMAChannel::new_client(
                    &config.stoc_channel_name,
                    config.buf_size,
                    config.sender_socket.parse().unwrap(),
                    1,
                )
                .unwrap();
                (0, Channel::new(Box::new(sender)), Channel::new(Box::new(receiver)))
            }
            &_ => panic!("Unsupported communication type in config"),
        };

        Self {
            id,
            channel_sender,
            channel_receiver,
            resource_idx: 0,
            #[cfg(feature = "local")]
            cuda_device: None,
        }
    }
}

impl Drop for ClientThread {
    fn drop(&mut self) {
        let proc_id = -1;
        proc_id.send(&self.channel_sender).unwrap();
    }
}

thread_local! {
    static CLIENT_THREAD: RefCell<ClientThread> = RefCell::new(ClientThread::new());
}

lazy_static! {
    static ref ELF_CONTROLLER: ElfController = ElfController::new();
}

#[ctor::ctor]
fn init() {
//     core_affinity::set_for_current(1);
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "debug");
    }
    env_logger::init();
    info!("[{}:{}] client init", std::file!(), std::line!());
}
