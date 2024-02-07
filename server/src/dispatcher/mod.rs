use super::*;

mod cudart_exe;
use self::cudart_exe::*;

pub fn dispatch(
    proc_id: i32,
    channel_sender: &mut RingBuffer<SHMChannelBufferManager>,
    channel_receiver: &mut RingBuffer<SHMChannelBufferManager>,
) {
    match proc_id {
        0 => cudaGetDeviceExe(channel_sender, channel_receiver),
        1 => cudaSetDeviceExe(channel_sender, channel_receiver),
        2 => cudaGetDeviceCountExe(channel_sender, channel_receiver),
        other => {
            error!("[{}:{}] invalid proc_id: {}", std::file!(), std::line!(), other);
        }
    }
}
