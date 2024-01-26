use std::time::Duration;

pub enum BufferPrivilege {
    BufferHost,
    BufferGuest,
}

pub enum IssuingMode {
    SyncIssuing,
    AsyncIssuing,
}

#[derive(Debug)]
pub enum DeviceBufferError {
    // Define error types, for example:
    IoError(std::io::Error),
    Timeout,
    // Add other relevant errors
}

const TIMEOUT: Duration = Duration::from_secs(50);

// Trait for the DeviceBuffer: an io abstraction for heterogenous device
pub trait DeviceBuffer {
    fn new(privilege: BufferPrivilege) -> Self
    where
        Self: Sized;

    fn put_bytes(&mut self, src: &[u8], mode: Option<IssuingMode>) -> Result<(), DeviceBufferError>;

    fn get_bytes(&mut self, dst: &mut [u8], mode: Option<IssuingMode>) -> Result<(), DeviceBufferError>;

    fn flush_out(&mut self, mode: Option<IssuingMode>) -> Result<(), DeviceBufferError>;

    fn fill_in(&mut self, mode: Option<IssuingMode>) -> Result<(), DeviceBufferError>;
}
