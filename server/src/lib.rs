extern crate device_buffer;
extern crate log;
mod cuda_lib;
mod dispatcher;

use cuda_lib::*;
use device_buffer::*;
use dispatcher::dispatch;

#[allow(unused_imports)]
use log::{debug, error, log_enabled, info, Level};

fn create_buffer() -> (SharedMemoryBuffer, SharedMemoryBuffer) {
    let buffer_sender =
        SharedMemoryBuffer::new(BufferPrivilege::BufferHost, SHM_NAME_STOC, SHM_BUFFER_SIZE)
            .unwrap();
    let buffer_receiver =
        SharedMemoryBuffer::new(BufferPrivilege::BufferHost, SHM_NAME_CTOS, SHM_BUFFER_SIZE)
            .unwrap();
    (buffer_sender, buffer_receiver)
}

fn receive_request(buffer_receiver: &SharedMemoryBuffer) -> Result<i32, DeviceBufferError> {
    let mut buf = [0u8; 4];
    if let Ok(4) = buffer_receiver.get_bytes(&mut buf, None) {
        Ok(i32::from_ne_bytes(buf))
    } else {
        Err(DeviceBufferError::IoError)
    }
}

pub fn launch_server() {
    let (buffer_sender, buffer_receiver) = create_buffer();
    info!("[{}:{}] shm buffer created", std::file!(), function!());
    if let cudaError_t::cudaSuccess = unsafe { cudaDeviceSynchronize() } {
        info!("[{}:{}] cuda driver initialized", std::file!(), function!());
    } else {
        error!("[{}:{}] failed to initialize cuda driver", std::file!(), function!());
        panic!();
    }

    loop {
        if let Ok(proc_id) = receive_request(&buffer_receiver) {
            dispatch(proc_id, &buffer_sender, &buffer_receiver);
        } else {
            error!("[{}:{}] failed to receive request", std::file!(), function!());
            break;
        }
    }

    info!("[{}:{}] server terminated", std::file!(), function!());
}
