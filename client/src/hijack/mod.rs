#![expect(non_snake_case)]

mod cuda_hijack;
mod cuda_hijack_custom;
mod cuda_unimplement;
mod cudart_hijack;
mod cudart_hijack_custom;
mod cudart_unimplement;
mod nvml_hijack;
mod nvml_unimplement;
mod cudnn_hijack;
mod cudnn_hijack_custom;
mod cudnn_unimplement;
mod cublas_hijack;
mod cublas_unimplement;
mod cublasLt_unimplement;

use codegen::{cuda_hook_hijack, use_thread_local};
use log::error;
use network::type_impl::{recv_slice_to, send_slice, MemPtr};
use network::{CommChannel, Transportable};

use crate::elf::interfaces::{fat_header, kernel_info_t};
use crate::{ClientThread, CLIENT_THREAD, ELF_CONTROLLER};
