#[macro_use]
extern crate lazy_static;

extern crate device_buffer;
use device_buffer::*;

pub mod cuda_runtime;
pub use cuda_runtime::*;

lazy_static! {
    static ref BUFFER_SENDER: SharedMemoryBuffer = {
        let buf =
            SharedMemoryBuffer::new(BufferPrivilege::BufferGuest, SHM_NAME_CTOS, SHM_BUFFER_SIZE)
                .unwrap();
        buf
    };
    static ref BUFFER_RECEIVER: SharedMemoryBuffer = {
        let buf =
            SharedMemoryBuffer::new(BufferPrivilege::BufferGuest, SHM_NAME_STOC, SHM_BUFFER_SIZE)
                .unwrap();
        buf
    };
}
