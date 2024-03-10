#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
use super::*;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, FromPrimitive, codegen::Transportable)]
#[allow(dead_code)]
pub enum nvmlReturn_enum {
    #[default]
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_MEMORY = 20,
    NVML_ERROR_NO_DATA = 21,
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24,
    NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25,
    NVML_ERROR_DEPRECATED = 26,
    NVML_ERROR_NOT_READY = 27,
    NVML_ERROR_UNKNOWN = 999,
}

pub use self::nvmlReturn_enum as nvmlReturn_t;

#[cfg(test)]
mod tests{
    use super::*;
    use crate::FromPrimitive;
    use crate::ringbufferchannel::{
        channel::META_AREA,
        LocalChannelBufferManager, RingBuffer
    };

    #[test]
    fn test_num_derive() {
        let x: u32 = nvmlReturn_t::NVML_SUCCESS as u32;
        assert_eq!(x, 0);
        match nvmlReturn_t::from_u32(1) {
            Some(v) => assert_eq!(v, nvmlReturn_t::NVML_ERROR_UNINITIALIZED),
            None => panic!("failed to convert from u32"),
        }
    }

    #[test]
    fn test_nvmlReturn_t_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let a = nvmlReturn_t::NVML_ERROR_UNINITIALIZED;
        let mut b = nvmlReturn_t::NVML_SUCCESS;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }
}
