use super::*;

mod cudart_exe;
use self::cudart_exe::*;

pub fn dispatch(
    proc_id: i32,
    buffer_sender: &SharedMemoryBuffer,
    buffer_receiver: &SharedMemoryBuffer,
) {
    match proc_id {
        0 => cudaGetDeviceExe(buffer_sender, buffer_receiver),
        1 => cudaSetDeviceExe(buffer_sender, buffer_receiver),
        2 => cudaGetDeviceCountExe(buffer_sender, buffer_receiver),
        other => {
            error!("[{}:{}] invalid proc_id: {}", std::file!(), function!(), other);
        }
    }
}
