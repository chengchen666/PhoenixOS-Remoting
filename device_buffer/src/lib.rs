extern crate num;
#[macro_use]
extern crate num_derive;
extern crate log;

use num::FromPrimitive;

pub use std::time::Duration;
pub mod shared_memory_buffer;
pub mod utils;

pub use utils::*;

#[allow(unused_imports)]
use log::{debug, error, log_enabled, info, Level};

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
    InvalidOperation,
    IoError,
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
pub use shared_memory_buffer::SHM_BUFFER_SIZE;
pub use shared_memory_buffer::SHM_NAME_STOC;
pub use shared_memory_buffer::SHM_NAME_CTOS;
