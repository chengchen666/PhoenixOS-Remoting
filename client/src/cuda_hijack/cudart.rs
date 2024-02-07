use super::*;

#[no_mangle]
pub extern "C" fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
    let proc_id = 0;
    let mut dev = 0;
    let mut result = cudaError_t::cudaSuccess;

    match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize proc_id: {:?}", e),
    }
    match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut dev) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize dev: {:?}", e),
    }
    match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
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
    println!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
    let proc_id = 1;
    let dev = device;
    let mut result = cudaError_t::cudaSuccess;
    match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize proc_id: {:?}", e),
    }
    match CHANNEL_SENDER.lock().unwrap().send_var(&dev) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize dev: {:?}", e),
    }
    match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize result: {:?}", e),
    }
    return result;
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaGetDeviceCount", std::file!(), std::line!());
    let proc_id = 2;
    let mut cnt = 0;
    let mut result = cudaError_t::cudaSuccess;
    match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
        Ok(_) => {}
        Err(e) => panic!("failed to serialize proc_id: {:?}", e),
    }
    match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut cnt) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize cnt: {:?}", e),
    }
    match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
        Ok(_) => {}
        Err(e) => panic!("failed to deserialize result: {:?}", e),
    }
    unsafe {
        *count = cnt;
    }
    return result;
}
