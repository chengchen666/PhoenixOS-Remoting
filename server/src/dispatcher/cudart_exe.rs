#![allow(non_snake_case)]
use super::*;

pub fn cudaGetDeviceExe(
    channel_sender: &mut RingBuffer<SHMChannelBufferManager>,
    _channel_receiver: &mut RingBuffer<SHMChannelBufferManager>,
) {
    info!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
    let mut device: i32 = 0;
    let result = unsafe { cudaGetDevice(&mut device) };
    channel_sender.send_var(&device).unwrap();
    channel_sender.send_var(&result).unwrap();
}

pub fn cudaSetDeviceExe(
    channel_sender: &mut RingBuffer<SHMChannelBufferManager>,
    channel_receiver: &mut RingBuffer<SHMChannelBufferManager>,
) {
    info!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
    let mut device: i32 = 0;
    channel_receiver.recv_var(&mut device).unwrap();
    let result = unsafe { cudaSetDevice(device) };
    channel_sender.send_var(&result).unwrap();
}

pub fn cudaGetDeviceCountExe(
    channel_sender: &mut RingBuffer<SHMChannelBufferManager>,
    _channel_receiver: &mut RingBuffer<SHMChannelBufferManager>,
) {
    info!("[{}:{}] cudaGetDeviceCount", std::file!(), std::line!());
    let mut count: i32 = 0;
    let result = unsafe { cudaGetDeviceCount(&mut count) };
    channel_sender.send_var(&count).unwrap();
    channel_sender.send_var(&result).unwrap();
}
