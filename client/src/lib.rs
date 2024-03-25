#[macro_use]
extern crate lazy_static;

extern crate cudasys;
extern crate network;

use cudasys::{
    cuda::{CUdevice, CUresult},
    cudart::{
        cudaDeviceProp, cudaError_t, cudaMemcpyKind, cudaStreamCaptureStatus, cudaStream_t,
        dim3,
    },
};
use network::{
    ringbufferchannel::{
        RingBuffer, SHMChannelBufferManager, SHM_NAME_CTOS, SHM_NAME_STOC, SHM_SIZE,
    },
    type_impl::{
        basic::MemPtr,
        nvml::nvmlReturn_t,
    },
    CommChannel, Transportable,
};

extern crate codegen;
use codegen::gen_hijack;

pub mod hijack;
pub use hijack::*;

pub mod elf;
use elf::interfaces::{fat_header, kernel_info_t};
use elf::ElfController;

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
    static ref ELF_CONTROLLER: ElfController = ElfController::new();
}
