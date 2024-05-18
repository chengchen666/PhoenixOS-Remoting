use super::*;
pub use crate::types::nvml::*;
include!("bindings/funcs/nvml.rs");

#[cfg(test)]
mod tests{
    use super::*;
    use crate::FromPrimitive;
    use network::ringbufferchannel::{
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
        let mut buffer: RingBuffer =
            RingBuffer::new(Box::new(LocalChannelBufferManager::new(10 + META_AREA)));
        let a = nvmlReturn_t::NVML_ERROR_UNINITIALIZED;
        let mut b = nvmlReturn_t::NVML_SUCCESS;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }
}
