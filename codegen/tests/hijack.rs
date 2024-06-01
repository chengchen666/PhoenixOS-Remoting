#![allow(non_snake_case)]
#[macro_use]
extern crate lazy_static;

extern crate log;
use log::info;

extern crate cudasys;
extern crate network;

use cudasys::cudart::cudaError_t;
use network::{
    ringbufferchannel::SHMChannel,
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

use codegen::gen_hijack;

gen_hijack!(0, "cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_hijack!(1, "cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
