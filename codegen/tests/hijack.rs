#![allow(non_snake_case)]
#[macro_use]
extern crate lazy_static;

extern crate network;
use network::{
    cudaError_t,
    ringbufferchannel::{
        RingBuffer, SHMChannelBufferManager, SHM_NAME_CTOS, SHM_NAME_STOC, SHM_SIZE,
    },
    CommChannel,
};

use std::sync::Mutex;

lazy_static! {
    static ref CHANNEL_SENDER: Mutex<RingBuffer<SHMChannelBufferManager>> = {
        let manager = SHMChannelBufferManager::new_client(SHM_NAME_CTOS, SHM_SIZE).unwrap();
        Mutex::new(RingBuffer::new(manager))
    };
    static ref CHANNEL_RECEIVER: Mutex<RingBuffer<SHMChannelBufferManager>> = {
        let manager = SHMChannelBufferManager::new_client(SHM_NAME_STOC, SHM_SIZE).unwrap();
        Mutex::new(RingBuffer::new(manager))
    };
}


extern crate codegen;

use codegen::gen_hijack;

gen_hijack!(0, "cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_hijack!(1, "cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
