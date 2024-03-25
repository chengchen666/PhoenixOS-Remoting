use super::*;
include!("bindings/cuda.rs");

#[cfg(test)]
mod tests{
    use super::*;
    use network::ringbufferchannel::{
        channel::META_AREA,
        LocalChannelBufferManager, RingBuffer
    };

    #[test]
    fn test_CUresult_io() {
        let mut buffer: RingBuffer<LocalChannelBufferManager> =
            RingBuffer::new(LocalChannelBufferManager::new(10 + META_AREA));
        let a = CUresult::CUDA_ERROR_ALREADY_ACQUIRED;
        let mut b = CUresult::CUDA_SUCCESS;
        a.send(&mut buffer).unwrap();
        b.recv(&mut buffer).unwrap();
        assert_eq!(a, b);
    }
}
