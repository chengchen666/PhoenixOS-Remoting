#[macro_use]
extern crate lazy_static;

extern crate device_buffer;
use device_buffer::*;

pub mod cuda_runtime;
pub use cuda_runtime::*;

lazy_static! {
    static ref BUF_SENDER: SharedMemoryBuffer = {
        let buf = SharedMemoryBuffer::new(BufferPrivilege::BufferHost, "/shm_buf", 1024).unwrap();
        buf
    };
}
