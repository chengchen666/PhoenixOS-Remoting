pub use std::time::Duration;
pub mod shared_memory_buffer;

pub enum BufferPrivilege {
    BufferHost,
    BufferGuest,
}

#[derive(Copy, Clone)]
pub enum IssuingMode {
    SyncIssuing,
    AsyncIssuing,
}

#[derive(Debug)]
pub enum DeviceBufferError {
    // Define error types, for example:
    IoError(std::io::Error),
    InvalidOperation,
    Timeout,
    // Add other relevant errors
}

// const BUFFER_TIMEOUT: Duration = Duration::from_secs(50);

// Trait for the DeviceBuffer: an io abstraction for heterogenous device
pub trait DeviceBuffer {
    fn put_bytes(&self, src: &[u8], mode: Option<IssuingMode>) -> Result<usize, DeviceBufferError>;

    fn get_bytes(&self, dst: &mut [u8], mode: Option<IssuingMode>) -> Result<usize, DeviceBufferError>;

    fn flush_out(&self, mode: Option<IssuingMode>) -> Result<(), DeviceBufferError>;

    fn fill_in(&self, mode: Option<IssuingMode>) -> Result<(), DeviceBufferError>;
}

pub use shared_memory_buffer::SharedMemoryBuffer;