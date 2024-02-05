#![allow(non_snake_case)]
use super::*;

pub fn cudaGetDeviceExe(buffer_sender: &SharedMemoryBuffer, _buffer_receiver: &SharedMemoryBuffer) {
    info!("[{}:{}] cudaGetDevice", std::file!(), function!());
    let mut device: i32 = 0;
    let result = unsafe { cudaGetDevice(&mut device) };
    serialize_i32(&device, buffer_sender).unwrap();
    serialize_cudaError_t(&result, buffer_sender).unwrap();
}

pub fn cudaSetDeviceExe(buffer_sender: &SharedMemoryBuffer, buffer_receiver: &SharedMemoryBuffer) {
    info!("[{}:{}] cudaSetDevice", std::file!(), function!());
    let mut device: i32 = 0;
    deserialize_i32(&mut device, buffer_receiver).unwrap();
    let result = unsafe { cudaSetDevice(device) };
    serialize_cudaError_t(&result, buffer_sender).unwrap();
}

pub fn cudaGetDeviceCountExe(
    buffer_sender: &SharedMemoryBuffer,
    _buffer_receiver: &SharedMemoryBuffer,
) {
    info!("[{}:{}] cudaGetDeviceCount", std::file!(), function!());
    let mut count: i32 = 0;
    let result = unsafe { cudaGetDeviceCount(&mut count) };
    serialize_i32(&count, buffer_sender).unwrap();
    serialize_cudaError_t(&result, buffer_sender).unwrap();
}
