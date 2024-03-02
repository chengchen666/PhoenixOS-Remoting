#![allow(non_snake_case)]

extern crate codegen;

use network::{Transportable, CommChannel, RawMemory, RawMemoryMut, CommChannelError};

#[repr(u32)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, codegen::Transportable)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub enum cudaError {
    #[default]
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
}
