use super::*;
include!("bindings/cudart.rs");

/// cudaStream_t is a pointer type, we just need to use usize to represent it.
/// It is not necessary to define a struct for it, as the struct is also just a placeholder.

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
        let x: u32 = cudaError_t::cudaSuccess as u32;
        assert_eq!(x, 0);
        match cudaError_t::from_u32(1) {
            Some(v) => assert_eq!(v, cudaError_t::cudaErrorInvalidValue),
            None => panic!("failed to convert from u32"),
        }
    }

    #[test]
    fn test_cudaError_t_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let a = cudaError_t::cudaErrorInvalidValue;
        let mut b = cudaError_t::cudaSuccess;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_cudaStream_t_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let a = 100usize as cudaStream_t;
        let mut b: cudaStream_t = 0usize as cudaStream_t;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    pub fn cuda_ffi() {
        let mut device = 0;
        let mut device_num = 0;

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDeviceCount(&mut device_num as *mut i32) }
        {
            println!("device count: {}", device_num);
        } else {
            panic!("failed to get device count");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, 0);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaSetDevice(device_num - 1) } {
        } else {
            panic!("failed to set device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, device_num - 1);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }
    }
}
