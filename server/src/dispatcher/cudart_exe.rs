#![allow(non_snake_case)]
use super::*;

gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_exe!("cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_exe!("cudaGetDeviceCount", "cudaError_t", "*mut ::std::os::raw::c_int");

// pub fn cudaGetDeviceExe<T: CommChannel>(channel_sender: &mut T, _channel_receiver: &mut T) {
//     info!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
//     let mut param1: ::std::os::raw::c_int = Default::default();
//     let result = unsafe { cudaGetDevice(&mut param1) };
//     param1.send(channel_sender).unwrap();
//     result.send(channel_sender).unwrap();
//     channel_sender.flush_out().unwrap();
// }

// pub fn cudaSetDeviceExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
//     info!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
//     let mut param1: ::std::os::raw::c_int = Default::default();
//     param1.recv(channel_receiver).unwrap();
//     let result = unsafe { cudaSetDevice(param1) };
//     result.send(channel_sender).unwrap();
//     channel_sender.flush_out().unwrap();
// }

// pub fn cudaGetDeviceCountExe<T: CommChannel>(channel_sender: &mut T, _channel_receiver: &mut T) {
//     info!("[{}:{}] cudaGetDeviceCount", std::file!(), std::line!());
//     let mut param1: ::std::os::raw::c_int = Default::default();
//     let result = unsafe { cudaGetDeviceCount(&mut param1) };
//     param1.send(channel_sender).unwrap();
//     result.send(channel_sender).unwrap();
//     channel_sender.flush_out().unwrap();
// }
