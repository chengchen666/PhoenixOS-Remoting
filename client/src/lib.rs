#[macro_use]
extern crate lazy_static;

extern crate network;
use network::{
    ringbufferchannel::{
        RingBuffer, SHMChannelBufferManager, SHM_NAME_CTOS, SHM_NAME_STOC, SHM_SIZE,
    },
    type_impl::cudart::{cudaError_t, cudaMemcpyKind, cudaStream_t},
    CommChannel, Transportable,
};

extern crate codegen;
use codegen::gen_hijack;

pub mod cuda_hijack;
pub use cuda_hijack::*;

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
