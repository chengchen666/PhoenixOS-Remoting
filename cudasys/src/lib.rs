#![allow(
    non_snake_case,
    non_upper_case_globals,
    non_camel_case_types,
    warnings,
    dead_code
)]

extern crate num;
pub use num::FromPrimitive;
#[macro_use]
extern crate num_derive;

extern crate network;
use network::{RawMemory, RawMemoryMut, CommChannel, CommChannelError, Transportable};
extern crate codegen;

pub mod cuda;

pub mod cudart;

#[cfg(test)]
mod tests {
    use super::*;

    // This should work without GPU
    #[test]
    fn get_version() {
        let mut version: i32 = 0;
        let result = unsafe { cudart::cudaDriverGetVersion(&mut version as *mut i32) };
        if result != cudart::cudaError::cudaSuccess {
            panic!("Cannot get driver version: ERROR={:?}", result);
        }
        println!("Version = {}", version);
    }
}
