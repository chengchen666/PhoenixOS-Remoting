#![expect(
    non_snake_case,
    non_upper_case_globals,
    non_camel_case_types,
    warnings,
    dead_code
)]

pub use num::FromPrimitive;
use num_derive::FromPrimitive;

use network::{RawMemory, RawMemoryMut, CommChannel, CommChannelError, Transportable};

// Type definitions extracted from the bindings.
pub mod types;

pub mod cuda;
pub mod cudart;
pub mod nvml;
pub mod cudnn;
pub mod cublas;
pub mod cublasLt;

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
