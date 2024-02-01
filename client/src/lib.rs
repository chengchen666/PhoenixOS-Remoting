#[macro_use]
extern crate lazy_static;

extern crate device_buffer;
use device_buffer::*;

pub mod cuda_runtime;
pub use cuda_runtime::*;

use std::sync::Mutex;

lazy_static! {
    static ref BUF_SENDER: Mutex<SharedMemoryBuffer> = {
        let buf = Mutex::new(SharedMemoryBuffer::new(BufferPrivilege::BufferHost, "/shm_buf", 1024).unwrap());
        buf
    };
}
