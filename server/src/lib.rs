extern crate device_buffer;
pub mod cuda_lib;
mod dispatcher;

pub use cuda_lib::*;
use device_buffer::*;

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
    if let Ok(4) = buffer_receiver.get_bytes(&mut buf, Some(IssuingMode::SyncIssuing)) {
        Ok(i32::from_le_bytes(buf))
    } else {
        Err(DeviceBufferError::IoError)
    }
}

pub fn launch_server() {
    let (buffer_sender, buffer_receiver) = create_buffer();
    println!("[{}:{}] shm buffer created", std::file!(), function!());
    unsafe {
        if let cudaError_t::cudaSuccess = cudaDeviceSynchronize() {
            println!("[{}:{}] cuda driver initialized", std::file!(), function!());
        } else {
            panic!("[{}:{}] failed to initialize cuda driver", std::file!(), function!());
        }
    }
    loop {
        if let Ok(proc_id) = receive_request(&buffer_receiver) {
            dispatcher::dispatch(proc_id, &buffer_sender, &buffer_receiver);
        } else {
            println!(
                "[{}:{}] failed to receive request",
                std::file!(),
                function!()
            );
            break;
        }
    }

    println!("[{}:{}] server terminated", std::file!(), function!());
}
