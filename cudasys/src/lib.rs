#![expect(
    non_snake_case,
)]

pub use num::FromPrimitive;
use num_derive::FromPrimitive;

// Type definitions extracted from the bindings.
pub mod types;

pub mod cuda;
pub mod cudart;
pub mod nvml;
pub mod cudnn;
pub mod cublas;
pub mod cublasLt;

mod hooks;

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
