use super::*;

mod cudart_exe;
use self::cudart_exe::*;

pub fn dispatch<T: CommChannel>(proc_id: i32, channel_sender: &mut T, channel_receiver: &mut T) {
    match proc_id {
        0 => cudaGetDeviceExe(channel_sender, channel_receiver),
        1 => cudaSetDeviceExe(channel_sender, channel_receiver),
        2 => cudaGetDeviceCountExe(channel_sender, channel_receiver),
        3 => cudaGetLastErrorExe(channel_sender, channel_receiver),
        4 => cudaPeekAtLastErrorExe(channel_sender, channel_receiver),
        5 => cudaStreamSynchronizeExe(channel_sender, channel_receiver),
        other => {
            error!(
                "[{}:{}] invalid proc_id: {}",
                std::file!(),
                std::line!(),
                other
            );
        }
    }
}
