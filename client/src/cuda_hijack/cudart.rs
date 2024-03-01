#![allow(non_snake_case)]
use super::*;

// gen_hijack!(0, "cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
// gen_hijack!(1, "cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
// gen_hijack!(2, "cudaGetDeviceCount", "cudaError_t", "*mut ::std::os::raw::c_int");

#[no_mangle]
pub extern "C" fn cudaGetDevice(param1: *mut ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
    let proc_id = 0;
    let mut var1: ::std::os::raw::c_int = Default::default();
    let mut result: cudaError_t = Default::default();

    match proc_id.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match CHANNEL_SENDER.lock().unwrap().flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    match var1.recv(&mut (*CHANNEL_RECEIVER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive var1: {:?}", e),
    }
    match result.recv(&mut (*CHANNEL_RECEIVER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    unsafe {
        *param1 = var1;
    }
    return result;
}

#[no_mangle]
pub extern "C" fn cudaSetDevice(param1: ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
    let proc_id = 1;
    let mut result: cudaError_t = Default::default();
    
    match proc_id.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match param1.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send param1: {:?}", e),
    }
    match CHANNEL_SENDER.lock().unwrap().flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    match result.recv(&mut (*CHANNEL_RECEIVER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    return result;
}

#[no_mangle]
pub extern "C" fn cudaGetDeviceCount(param1: *mut ::std::os::raw::c_int) -> cudaError_t {
    println!("[{}:{}] cudaGetDeviceCount", std::file!(), std::line!());
    let proc_id = 2;
    let mut var1: ::std::os::raw::c_int = Default::default();
    let mut result: cudaError_t = Default::default();

    match proc_id.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match CHANNEL_SENDER.lock().unwrap().flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    match var1.recv(&mut (*CHANNEL_RECEIVER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive var1: {:?}", e),
    }
    match result.recv(&mut (*CHANNEL_RECEIVER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    unsafe {
        *param1 = var1;
    }
    return result;
}
