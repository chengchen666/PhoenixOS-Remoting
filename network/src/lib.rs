use std::error::Error;
use std::fmt;

extern crate num;
pub use num::FromPrimitive;
#[macro_use]
extern crate num_derive;

pub mod buffer;
pub use buffer::{BufferError, RawBuffer};

pub mod ringbufferchannel;
pub mod type_impl;

pub use type_impl::cudaError_t;

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

    fn send_var<T: SerializeAndDeserialize>(&mut self, value: &T) -> Result<(), CommChannelError> {
        let buf = value.to_bytes()?;
        let len = self.send(&buf)?;
        if len != buf.len() {
            return Err(CommChannelError::IoError);
        }
        Ok(())
    }

    fn recv_var<T: SerializeAndDeserialize>(&mut self, value: &mut T) -> Result<(), CommChannelError> {
        let mut buf = vec![0u8; std::mem::size_of::<T>()];
        let len = self.recv(&mut buf)?;
        value.from_bytes(&buf[0..len].to_vec())
    }
}

///
/// The type can be transfered by the channel
pub trait SerializeAndDeserialize{
    /// Serialize the type to bytes
    fn to_bytes(&self) -> Result<Vec<u8>, CommChannelError>;

    /// Deserialize the type from bytes
    fn from_bytes(&mut self, src: &Vec<u8>) -> Result<(), CommChannelError>;
}
