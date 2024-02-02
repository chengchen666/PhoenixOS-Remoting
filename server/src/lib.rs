extern crate device_buffer;
pub mod cuda_lib;
mod dispatcher;

pub use cuda_lib::*;
use device_buffer::*;
use dispatcher::dispatch;

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
    if let cudaError_t::cudaSuccess = unsafe { cudaDeviceSynchronize() } {
        println!("[{}:{}] cuda driver initialized", std::file!(), function!());
    } else {
        panic!(
            "[{}:{}] failed to initialize cuda driver",
            std::file!(),
            function!()
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn cuda_ffi() {
        let mut device = 0;
        let mut device_num = 0;

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDeviceCount(&mut device_num as *mut i32) }
        {
            println!("device count: {}", device_num);
        } else {
            panic!("failed to get device count");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, 0);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaSetDevice(device_num - 1) } {
        } else {
            panic!("failed to set device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, device_num - 1);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }
    }
}
