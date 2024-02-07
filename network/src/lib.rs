use std::error::Error;
use std::fmt;

pub mod buffer;
pub use buffer::{BufferError, RawBuffer};

pub mod ringbufferchannel;

#[derive(Debug)]
pub enum CommChannelError {
    // Define error types, for example:
    InvalidOperation,
    IoError,
    Timeout,
    NoLeftSpace,
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
    fn send(&mut self, src: &[u8]) -> Result<usize, CommChannelError>;

    /// Non-block version
    /// Immediately return if error such as no left space
    fn try_send(&mut self, src: &[u8]) -> Result<usize, CommChannelError>;

    /// Read bytes from the channel
    /// Wait if dont receive enough bytes
    fn recv(&mut self, dst: &mut [u8]) -> Result<usize, CommChannelError>;

    /// Non-block version
    /// Immediately return after receive however long bytes (maybe =0 or <len)
    fn try_recv(&mut self, dst: &mut [u8]) -> Result<usize, CommChannelError>;

    /// Flush the all the buffered results to the channel 
    fn flush_out(&mut self) -> Result<(), CommChannelError>;
}
