use super::*;

// TODO: use device_buffer for communication

#[no_mangle]
pub extern "C" fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaGetDevice", std::file!(), function!());
    let proc_id = 0;
    let mut dev = 0;
    let mut result = cudaError_t::cudaSuccess;
    match serialize_i32(&proc_id, &BUFFER_SENDER) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize proc_id: {:?}", e),
    }
    match deserialize_i32(&mut dev, &BUFFER_SENDER) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize dev: {:?}", e),
    }
    match deserialize_cudaError_t(&mut result, &BUFFER_RECEIVER) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize result: {:?}", e),
    }
    unsafe {
        *device = dev;
    }
    return result;
}

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaSetDevice", std::file!(), function!());
    let proc_id = 1;
    let dev = device;
    let mut result = cudaError_t::cudaSuccess;
    match serialize_i32(&proc_id, &BUFFER_SENDER) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize proc_id: {:?}", e),
    }
    match serialize_i32(&dev, &BUFFER_SENDER) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize dev: {:?}", e),
    }
    match deserialize_cudaError_t(&mut result, &BUFFER_RECEIVER) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize result: {:?}", e),
    }
    return result;
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaGetDeviceCount", std::file!(), function!());
    let proc_id = 2;
    let mut cnt = 0;
    let mut result = cudaError_t::cudaSuccess;
    match serialize_i32(&proc_id, &BUFFER_SENDER) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize proc_id: {:?}", e),
    }
    match deserialize_i32(&mut cnt, &BUFFER_SENDER) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize cnt: {:?}", e),
    }
    match deserialize_cudaError_t(&mut result, &BUFFER_RECEIVER) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize result: {:?}", e),
    }
    unsafe {
        *count = cnt;
    }
    return result;
}
