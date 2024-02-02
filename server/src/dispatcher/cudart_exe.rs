#![allow(non_snake_case)]
use super::*;

// TODO: implement to_ne_bytes for cudaError_t

pub fn cudaGetDeviceExe(buffer_sender: &SharedMemoryBuffer, buffer_receiver: &SharedMemoryBuffer) {
    println!("[{}:{}] cudaGetDevice", std::file!(), function!());
    let mut device: i32 = 0;
    let result = unsafe { cudaGetDevice(&mut device) };
    buffer_sender.put_bytes(&device.to_ne_bytes(), None).unwrap();
    // buffer_sender.put_bytes(&result.to_ne_bytes(), None).unwrap();
}

pub fn cudaSetDeviceExe(buffer_sender: &SharedMemoryBuffer, buffer_receiver: &SharedMemoryBuffer) {
    println!("[{}:{}] cudaSetDevice", std::file!(), function!());
    let mut device: i32 = 0;
    buffer_receiver.get_bytes(&mut device.to_ne_bytes(), None).unwrap();
    let result = unsafe { cudaSetDevice(device) };
    // buffer_sender.put_bytes(&result.to_ne_bytes(), None).unwrap();
}

pub fn cudaGetDeviceCountExe(buffer_sender: &SharedMemoryBuffer, buffer_receiver: &SharedMemoryBuffer) {
    println!("[{}:{}] cudaGetDeviceCount", std::file!(), function!());
    let mut count: i32 = 0;
    let result = unsafe { cudaGetDeviceCount(&mut count) };
    buffer_sender.put_bytes(&count.to_ne_bytes(), None).unwrap();
    // buffer_sender.put_bytes(&result.to_ne_bytes(), None).unwrap();
}
