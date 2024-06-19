#![allow(non_snake_case)]
extern crate lazy_static;
use lazy_static::lazy_static;

extern crate cudasys;
extern crate network;

extern crate log;
use log::info;

use cudasys::cudart::cudaError_t;
use network::{
    ringbufferchannel::SHMChannel,
    type_impl::MemPtr,
    Channel, CommChannel, Transportable
};

use std::sync::Mutex;
use std::boxed::Box;

lazy_static! {
    static ref CHANNEL_SENDER: Mutex<Channel> = {
        let c = Box::new(SHMChannel::new_client("/ctos", 104857520).unwrap());
        Mutex::new(Channel::new(c))
    };
    static ref CHANNEL_RECEIVER: Mutex<Channel> = {
        let c = Box::new(SHMChannel::new_client("/stoc", 104857520).unwrap());
        Mutex::new(Channel::new(c))
    };
    static ref ENABLE_LOG: bool = {
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "debug");
        }
        env_logger::init();
        true
    };
}

extern crate codegen;

use codegen::{gen_hijack_async, gen_exe_async};

gen_hijack_async!(8, "cudaFree", "cudaError_t", "MemPtr");