#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
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

    /// Send a variable to the channel and *flush*
    fn send_var<T: SerializeAndDeserialize>(&mut self, value: &T) -> Result<(), CommChannelError>
    where
        [(); std::mem::size_of::<T>()]:,
    {
        let buf = value.to_bytes()?;
        let len = self.send(&buf)?;
        if len != buf.len() {
            return Err(CommChannelError::IoError);
        }
        self.flush_out()?;
        Ok(())
    }

    /// Receive a variable from the channel
    fn recv_var<T: SerializeAndDeserialize>(
        &mut self,
        value: &mut T,
    ) -> Result<(), CommChannelError>
    where
        [(); std::mem::size_of::<T>()]:,
    {
        let mut buf = [0u8; std::mem::size_of::<T>()];
        let len = self.recv(&mut buf)?;
        value.from_bytes(&buf[0..len])
    }
}

///
/// The type can be transfered by the channel
pub trait SerializeAndDeserialize: Sized {
    /// TODO: compare with arena
    
    /// Serialize the type to bytes
    fn to_bytes(&self) -> Result<[u8; std::mem::size_of::<Self>()], CommChannelError>;

    /// Deserialize the type from bytes
    fn from_bytes(&mut self, src: &[u8]) -> Result<(), CommChannelError>;
}
