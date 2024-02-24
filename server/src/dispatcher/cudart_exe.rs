#![allow(non_snake_case)]
use super::*;

pub fn cudaGetDeviceExe<T: CommChannel>(channel_sender: &mut T, _channel_receiver: &mut T) {
    info!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
    let mut param1: ::std::os::raw::c_int = 0;
    let result = unsafe { cudaGetDevice(&mut param1) };
    channel_sender.send_var(&param1).unwrap();
    channel_sender.send_var(&result).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudaSetDeviceExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
    let mut param1: ::std::os::raw::c_int = 0;
    channel_receiver.recv_var(&mut param1).unwrap();
    let result = unsafe { cudaSetDevice(param1) };
    channel_sender.send_var(&result).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudaGetDeviceCountExe<T: CommChannel>(channel_sender: &mut T, _channel_receiver: &mut T) {
    info!("[{}:{}] cudaGetDeviceCount", std::file!(), std::line!());
    let mut param1: ::std::os::raw::c_int = 0;
    let result = unsafe { cudaGetDeviceCount(&mut param1) };
    channel_sender.send_var(&param1).unwrap();
    channel_sender.send_var(&result).unwrap();
    channel_sender.flush_out().unwrap();
}
