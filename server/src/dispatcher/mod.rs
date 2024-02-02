use super::*;
mod cudart;

use cudart::*;

pub fn dispatch(
    proc_id: i32,
    buffer_sender: &SharedMemoryBuffer,
    buffer_receiver: &SharedMemoryBuffer,
) {
    match proc_id {
        0 => cudaGetDeviceExe(buffer_sender, buffer_receiver),
        1 => cudaSetDeviceExe(buffer_sender, buffer_receiver),
        other => {
            println!(
                "[{}:{}] invalid proc_id: {}",
                std::file!(),
                function!(),
                other
            );
        }
    }
}
