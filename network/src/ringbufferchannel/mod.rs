pub mod channel;
pub use channel::RingBuffer;
pub mod test;

// Only implemented in Linux now
#[cfg(target_os = "linux")]
pub mod shm;

/// A ring buffer can use arbitrary memory for its channel 
/// 
/// It will manage the following:
/// - The buffer memory allocation and 
/// - The buffer memory deallocation
pub trait ChannelBufferManager { 
    fn get_managed_memory(&self) -> (*mut u8, usize);
}

/// A simple local channel buffer manager 
pub struct LocalChannelBufferManager {
    buffer: Vec<u8>,
}

impl LocalChannelBufferManager {
    pub fn new(size: usize) -> LocalChannelBufferManager {
        LocalChannelBufferManager {
            buffer: vec![0; size],
        }
    }
}

impl ChannelBufferManager for LocalChannelBufferManager {
    fn get_managed_memory(&self) -> (*mut u8, usize) {
        (self.buffer.as_ptr() as *mut u8, self.buffer.len())
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
        assert_eq!(len, size);
        unsafe {
            assert_eq!(*ptr, 0);
            assert_eq!(*ptr.add(size - 1), 0);
        }
    }
}


