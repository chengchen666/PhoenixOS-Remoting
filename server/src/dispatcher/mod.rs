use super::*;

mod cudart_exe;
mod cudart_exe_custom;
mod nvml_exe;

use self::cudart_exe::*;
use self::cudart_exe_custom::*;
use self::nvml_exe::*;

pub fn dispatch<T: CommChannel>(proc_id: i32, channel_sender: &mut T, channel_receiver: &mut T) {
    match proc_id {
        0 => cudaGetDeviceExe(channel_sender, channel_receiver),
        1 => cudaSetDeviceExe(channel_sender, channel_receiver),
        2 => cudaGetDeviceCountExe(channel_sender, channel_receiver),
        3 => cudaGetLastErrorExe(channel_sender, channel_receiver),
        4 => cudaPeekAtLastErrorExe(channel_sender, channel_receiver),
        5 => cudaStreamSynchronizeExe(channel_sender, channel_receiver),
        6 => cudaMallocExe(channel_sender, channel_receiver),
        7 => cudaMemcpyExe(channel_sender, channel_receiver),
        8 => cudaFreeExe(channel_sender, channel_receiver),
        9 => cudaStreamIsCapturingExe(channel_sender, channel_receiver),
        10 => cudaGetDevicePropertiesExe(channel_sender, channel_receiver),
        1000 => nvmlInit_v2Exe(channel_sender, channel_receiver),
        1001 => nvmlDeviceGetCount_v2Exe(channel_sender, channel_receiver),
        1002 => nvmlInitWithFlagsExe(channel_sender, channel_receiver),
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
