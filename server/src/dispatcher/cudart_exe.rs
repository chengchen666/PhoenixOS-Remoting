#![allow(non_snake_case)]
use super::*;
use num::FromPrimitive;

// TODO: implement (de-)serialization for basic types

fn serialize_i32(value: &i32, buffer: &SharedMemoryBuffer) -> Result<usize, DeviceBufferError> {
    let buf = value.to_ne_bytes();
    buffer.put_bytes(&buf, None)
}

fn deserialize_i32(
    value: &mut i32,
    buffer: &SharedMemoryBuffer,
) -> Result<usize, DeviceBufferError> {
    let mut buf = [0u8; std::mem::size_of::<i32>()];
    let len_read = buffer.get_bytes(&mut buf, None)?;
    *value = i32::from_ne_bytes(buf);
    Ok(len_read)
}

fn serialize_cudaError_t(
    value: &cudaError_t,
    buffer: &SharedMemoryBuffer,
) -> Result<usize, DeviceBufferError> {
    let buf = (*value as u32).to_ne_bytes();
    buffer.put_bytes(&buf, None)
}

fn deserialize_cudaError_t(
    value: &mut cudaError_t,
    buffer: &SharedMemoryBuffer,
) -> Result<usize, DeviceBufferError> {
    let mut buf = [0u8; std::mem::size_of::<cudaError_t>()];
    let len_read = buffer.get_bytes(&mut buf, None)?;
    match cudaError_t::from_u32(u32::from_ne_bytes(buf)) {
        Some(v) => *value = v,
        None => return Err(DeviceBufferError::IoError),
    }
    Ok(len_read)
}

pub fn cudaGetDeviceExe(buffer_sender: &SharedMemoryBuffer, _buffer_receiver: &SharedMemoryBuffer) {
    println!("[{}:{}] cudaGetDevice", std::file!(), function!());
    let mut device: i32 = 0;
    let result = unsafe { cudaGetDevice(&mut device) };
    serialize_i32(&device, buffer_sender).unwrap();
    serialize_cudaError_t(&result, buffer_sender).unwrap();
}

pub fn cudaSetDeviceExe(buffer_sender: &SharedMemoryBuffer, buffer_receiver: &SharedMemoryBuffer) {
    println!("[{}:{}] cudaSetDevice", std::file!(), function!());
    let mut device: i32 = 0;
    deserialize_i32(&mut device, buffer_receiver).unwrap();
    let result = unsafe { cudaSetDevice(device) };
    serialize_cudaError_t(&result, buffer_sender).unwrap();
}

pub fn cudaGetDeviceCountExe(
    buffer_sender: &SharedMemoryBuffer,
    _buffer_receiver: &SharedMemoryBuffer,
) {
    println!("[{}:{}] cudaGetDeviceCount", std::file!(), function!());
    let mut count: i32 = 0;
    let result = unsafe { cudaGetDeviceCount(&mut count) };
    serialize_i32(&count, buffer_sender).unwrap();
    serialize_cudaError_t(&result, buffer_sender).unwrap();
}
