use std::error::Error;
use std::{fmt, mem, ptr, slice};

pub mod buffer;
pub use buffer::{RawBuffer,BufferError};

#[derive(Debug)]
pub enum CommChannelError {
    // Define error types, for example:
    InvalidOperation,
    IoError,
    Timeout,
    // Add other relevant errors
}

impl fmt::Display for CommChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: refine
        write!(f, "Channel Error: {:?}", self)
    }
}

impl Error for CommChannelError {}


///
/// A communication channel allows TBD
pub trait CommChannel { 
    /// Write bytes to the channel
    /// It may flush if the channel has no left space
    fn put_bytes(&mut self, src: &[u8]) -> Result<usize, CommChannelError>;

    /// Read bytes from the channel
    fn get_bytes(&mut self, dst: &mut [u8]) -> Result<usize, CommChannelError>;

    /// Flush the all the buffered results to the channel 
    fn flush_out(&mut self) -> Result<(), CommChannelError>;
}