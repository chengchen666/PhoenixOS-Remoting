#[macro_use]
extern crate lazy_static;

extern crate cudasys;
extern crate network;

extern crate log;
#[allow(unused_imports)]
use log::{debug, error, info, log_enabled, Level};

#[allow(unused_imports)]
use network::{
    ringbufferchannel::{RDMAChannel, SHMChannel},
    type_impl::MemPtr,
    Channel, CommChannel, Transportable, CONFIG,
};

extern crate codegen;
use codegen::gen_hijack;
use codegen::gen_unimplement;

pub mod hijack;
pub use hijack::*;

pub mod elf;
use elf::interfaces::{fat_header, kernel_info_t};
use elf::ElfController;

pub mod dl;
pub use dl::*;

use std::boxed::Box;
use std::sync::Mutex;

lazy_static! {
    // Use features when compiling to decide what arm(s) will be supported.
    // In the client side, the sender's name is ctos_channel_name,
    // receiver's name is stoc_channel_name.
    static ref CHANNEL_SENDER: Mutex<Channel> = {
        match CONFIG.comm_type.as_str() {
            #[cfg(feature = "shm")]
            "shm" => {
                let c = SHMChannel::new_client(&CONFIG.ctos_channel_name, CONFIG.buf_size).unwrap();
                Mutex::new(Channel::new(Box::new(c)))
            },
            #[cfg(feature = "rdma")]
            "rdma" => {
                // client side sender should connect to server's receiver socket.
                let c = RDMAChannel::new_client(&CONFIG.ctos_channel_name, CONFIG.buf_size, CONFIG.receiver_socket.parse().unwrap(), 1).unwrap();
                Mutex::new(Channel::new(Box::new(c)))
            }
            &_ => panic!("Unsupported communication type in config"),
        }
    };
    static ref CHANNEL_RECEIVER: Mutex<Channel> = {
        match CONFIG.comm_type.as_str() {
            #[cfg(feature = "shm")]
            "shm" => {
                let c = SHMChannel::new_client(&CONFIG.stoc_channel_name, CONFIG.buf_size).unwrap();
                Mutex::new(Channel::new(Box::new(c)))
            }
            #[cfg(feature = "rdma")]
            "rdma" => {
                // client side receiver should connect to server's sender socket.
                let c = RDMAChannel::new_client(&CONFIG.stoc_channel_name, CONFIG.buf_size, CONFIG.sender_socket.parse().unwrap(), 1).unwrap();
                Mutex::new(Channel::new(Box::new(c)))
            }
            &_ => panic!("Unsupported communication type in config"),
        }
    };

    static ref ELF_CONTROLLER: ElfController = ElfController::new();
    static ref ENABLE_LOG: bool = {
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "debug");
        }
        env_logger::init();
        true
    };
}
