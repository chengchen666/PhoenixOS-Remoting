#![expect(incomplete_features)]
use serde::Deserialize;
use std::error::Error;
use std::boxed::Box;
use std::fmt;

use lazy_static::lazy_static;

pub mod ringbufferchannel;
pub mod type_impl;

pub use ringbufferchannel::types::NsTimestamp;

#[derive(Deserialize)]
pub struct NetworkConfig {
    pub comm_type: String,
    pub sender_socket: String,
    pub receiver_socket: String,
    pub daemon_socket: String,
    pub stoc_channel_name: String,
    pub ctos_channel_name: String,
    pub buf_size: usize,
    pub rtt: f64,
    pub bandwidth: f64,
}


lazy_static! {
    pub static ref CONFIG: NetworkConfig = {
        // Use environment variable to set config file's path.
        let path = match std::env::var("NETWORK_CONFIG") {
            Ok(val) => val,
            Err(_) => concat!(env!("CARGO_MANIFEST_DIR"), "/../config.toml").to_owned(),
        };
        let content = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
        toml::from_str(&content).expect("Failed to parse config.toml")
    };
}

/// A raw memory struct
/// used to wrap raw memory pointer and length,
/// for brevity of `CommChannel` interface
pub struct RawMemory {
    pub ptr: *const u8,
    pub len: usize,
}

impl RawMemory {
    pub fn new<T>(var: &T, len: usize) -> Self {
        RawMemory {
            ptr: var as *const T as *const u8,
            len,
        }
    }

    pub fn from_ptr(ptr: *const u8, len: usize) -> Self {
        RawMemory { ptr, len }
    }

    pub fn add_offset(&self, offset: usize) -> Self {
        RawMemory {
            ptr: unsafe { self.ptr.add(offset) },
            len: self.len - offset,
        }
    }

    pub fn as_slice(&self) -> Option<&[u8]> {
        unsafe {
            if self.ptr.is_null() || self.len == 0 {
                None
            } else {
                Some(std::slice::from_raw_parts(self.ptr, self.len))
            }
        }
    }
}

/// A *mutable* raw memory struct
/// used to wrap raw memory pointer and length,
/// for brevity of `CommChannel` interface
pub struct RawMemoryMut {
    pub ptr: *mut u8,
    pub len: usize,
}

impl RawMemoryMut {
    pub fn new<T>(var: &mut T, len: usize) -> Self {
        RawMemoryMut {
            ptr: var as *mut T as *mut u8,
            len,
        }
    }

    pub fn from_ptr(ptr: *mut u8, len: usize) -> Self {
        RawMemoryMut { ptr, len }
    }

    pub fn add_offset(&self, offset: usize) -> Self {
        RawMemoryMut {
            ptr: unsafe { self.ptr.add(offset) },
            len: self.len - offset,
        }
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        unsafe {
            if self.ptr.is_null() || self.len == 0 {
                None
            } else {
                Some(std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len))
            }
        }
    }
}

#[derive(Debug)]
pub enum CommChannelError {
    // Define error types, for example:
    InvalidOperation,
    IoError,
    Timeout,
    NoLeftSpace,
    BlockOperation,
    // Add other relevant errors
}

impl fmt::Display for CommChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: refine
        write!(f, "Channel Error: {:?}", self)
    }
}

impl Error for CommChannelError {}

pub use CommChannelInner as CommChannel;

impl CommChannelInnerIO for Channel {
    /// Write bytes to the channel
    /// It may flush if the channel has no left space
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        self.get_inner().put_bytes(src)
    }

    /// Non-block version
    /// Immediately return if error such as no left space
    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError> {
        self.get_inner().try_put_bytes(src)
    }

    /// Read bytes from the channel
    /// Wait if dont receive enough bytes
    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_inner().get_bytes(dst)
    }

    /// Non-block version
    /// Immediately return after receive however long bytes (maybe =0 or <len)
    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_inner().try_get_bytes(dst)
    }

    /// Non-block versoin
    /// Return immediately if there's not enough bytes in channel
    fn safe_try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError> {
        self.get_inner().safe_try_get_bytes(dst)
    }
}

impl CommChannelInner for Channel {
    /// Flush the all the buffered results to the channel
    fn flush_out(&self) -> Result<(), CommChannelError> {
        self.get_inner().flush_out()
    }

    /// Only for emulator channel
    /// Will recv timestamp and busy waiting until it's time to receive
    fn recv_ts(&self) -> Result<(), CommChannelError> {
        self.get_inner().recv_ts()
    }
}

///
/// A communication channel allows TBD
pub struct Channel {
    inner: Box<dyn CommChannelInner>,
}

impl Channel {
    pub fn new(inner: Box<dyn CommChannelInner>) -> Self {
        Self {
            inner,
        }
    }

    fn get_inner(&self) -> &Box<dyn CommChannelInner> {
        &self.inner
    }
}

/// communication interface
pub trait CommChannelInner: CommChannelInnerIO + Send + Sync {
    fn flush_out(&self) -> Result<(), CommChannelError>;

    fn recv_ts(&self) -> Result<(), CommChannelError> {
        Ok(())
    }
}

pub trait CommChannelInnerIO {
    fn put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError>;

    fn try_put_bytes(&self, src: &RawMemory) -> Result<usize, CommChannelError>;

    fn get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;

    fn try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;

    fn safe_try_get_bytes(&self, dst: &mut RawMemoryMut) -> Result<usize, CommChannelError>;
}

///
/// The type itself use `CommChannel` to implicitly implement (de-)serialization logic.
///
/// Every type wanted to be transfered should implement this trait.
pub trait Transportable {
    fn send<C: CommChannel>(&self, channel: &C) -> Result<(), CommChannelError>;

    fn recv<C: CommChannel>(&mut self, channel: &C) -> Result<(), CommChannelError>;
}

pub trait TransportableMarker: Copy {}
