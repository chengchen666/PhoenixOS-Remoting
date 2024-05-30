pub mod channel;
use std::ptr::NonNull;

pub use channel::RingBuffer;
pub mod test;

// Only implemented in Linux now
#[cfg(target_os = "linux")]
pub mod shm;
#[cfg(target_os = "linux")]
pub use shm::SHMChannelBufferManager;

pub mod rdma;
pub use rdma::RDMAChannelBufferManager;

pub mod utils;

pub mod emulator;
pub use emulator::EmulatorBuffer;
pub mod types;
pub use types::*;

pub const CACHE_LINE_SZ: usize = 64;

/// A ring buffer can use arbitrary memory for its channel
///
/// It will manage the following:
/// - The buffer memory allocation and
/// - The buffer memory deallocation
pub trait ChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize);
    fn read_at(&self, offset: usize, dst: *mut u8, count: usize) -> usize;
    fn write_at(&self, offset: usize, src: *const u8, count: usize) -> usize;
}

/// A simple local channel buffer manager
pub struct LocalChannelBufferManager {
    buffer: NonNull<u8>,
    size: usize,
}

unsafe impl Send for LocalChannelBufferManager {}

impl Drop for LocalChannelBufferManager {
    fn drop(&mut self) {
        utils::deallocate(self.buffer, self.size, CACHE_LINE_SZ);
    }
}

impl LocalChannelBufferManager {
    pub fn new(size: usize) -> LocalChannelBufferManager {
        LocalChannelBufferManager {
            buffer: utils::allocate_cache_line_aligned(size, CACHE_LINE_SZ),
            size: size,
        }
    }
}

impl ChannelBufferManager for LocalChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.buffer.as_ptr(), self.size)
    }

    fn read_at(&self, offset: usize, dst: *mut u8, count: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(self.buffer.as_ptr().add(offset) as _, dst, count);
        }
        count
    }

    fn write_at(&self, offset: usize, src: *const u8, count: usize) -> usize {
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.buffer.as_ptr().add(offset) as _, count);
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_channel_buffer_manager() {
        let size = 64;
        let manager = LocalChannelBufferManager::new(size);
        let (ptr, len) = manager.get_managed_memory();
        assert!(utils::is_cache_line_aligned(ptr));

        assert_eq!(len, size);
    }
}
