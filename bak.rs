#![allow(non_snake_case)]
use super::*;
use cudasys::types::cudart::*;
use ::std::os::raw::*;

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    assert_eq!(true, *ENABLE_LOG);
    info!("[{}:{}] cudaMemcpy", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    if cudaMemcpyKind::cudaMemcpyHostToHost == kind {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
        }
        return cudaError_t::cudaSuccess;
    }

    let proc_id = 7;
    let mut result: cudaError_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match dst.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send dst: {:?}", e),
    }
    match src.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send src: {:?}", e),
    }
    match count.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send count: {:?}", e),
    }
    match kind.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send kind: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        // transport [src; count] to device
        let data = unsafe { std::slice::from_raw_parts(src as *const u8, count) };
        match data.send(channel_sender) {
            Ok(()) => {}
            Err(e) => panic!("failed to send data: {:?}", e),
        }
    }

    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        // receive [dst; count] from device
        let data = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, count) };
        match data.recv(channel_receiver) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive data: {:?}", e),
        }
    }

    #[cfg(feature = "async_api")]
    {
        return cudaError_t::cudaSuccess;
    }
    #[cfg(not(feature = "async_api"))]
    {
        match result.recv(channel_receiver) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive result: {:?}", e),
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        return result;
    }
}

// TODO: maybe we should understand the semantic diff of cudaMemcpyAsync&cudaMemcpy
#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
    _stream: cudaStream_t,
) -> cudaError_t {
    assert_eq!(true, *ENABLE_LOG);
    info!("[{}:{}] cudaMemcpyAsync", std::file!(), std::line!());
    cudaMemcpy(dst, src, count, kind)
}

use std::arch::asm;
#[feature(asm)]
#[inline]
fn rdtscp() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        asm!("rdtscp", out("rax") lo, out("rdx") hi, options(nomem, nostack));
        ((hi as u64) << 32) | (lo as u64)
    }
}
fn clock2ns(clock: u64) -> f64 {
    clock as f64 / 2.5
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    func: MemPtr,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    assert_eq!(true, *ENABLE_LOG);
    info!("[{}:{}] cudaLaunchKernel", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    let proc_id = 200;
    let mut result: cudaError_t = Default::default();

    let total_start = rdtscp();
    let serialize_start = rdtscp();
    let info: *mut kernel_info_t =
        ELF_CONTROLLER.find_kernel_host_func(func as *mut ::std::os::raw::c_void);
    if info.is_null() {
        panic!("request to call unknown kernel.");
    }
    let info = unsafe { &mut *info };

    let argc = info.param_num;
    let mut arg_vec: Vec<Vec<u8>> = Vec::new();
    for i in 0..argc {
        let size = unsafe { *info.param_sizes.wrapping_add(i) as usize };
        let arg: Vec<u8> =
            unsafe { std::slice::from_raw_parts((*args.add(i)) as *const u8, size).to_vec() };
        arg_vec.push(arg);
    }

    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match func.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send func: {:?}", e),
    }
    match gridDim.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send gridDim: {:?}", e),
    }
    match blockDim.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send blockDim: {:?}", e),
    }
    match argc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send argc: {:?}", e),
    }
    for arg in arg_vec.iter() {
        match arg.send(channel_sender) {
            Ok(()) => {}
            Err(e) => panic!("failed to send arg: {:?}", e),
        }
    }
    match sharedMem.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send sharedMem: {:?}", e),
    }
    match stream.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send stream: {:?}", e),
    }
    let serialize_end = rdtscp();
    let send_start = rdtscp();
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    let send_end = rdtscp();

    #[cfg(feature = "async_api")]
    {
        let total_end = rdtscp();
        println!("total_time: {}\nserialize_time: {}\nsend_time: {}\n",
            clock2ns(total_end - total_start),
            clock2ns(serialize_end - serialize_start),
            clock2ns(send_end - send_start));
        return cudaError_t::cudaSuccess;
    }
    #[cfg(not(feature = "async_api"))]
    {
        let recv_start = rdtscp();
        match result.recv(channel_receiver) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive result: {:?}", e),
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        let recv_end = rdtscp();
        let total_end = rdtscp();
        println!("total_time: {}\nserialize_time: {}\nsend_time: {}\nrecv_time: {}\n",
            clock2ns(total_end - total_start),
            clock2ns(serialize_end - serialize_start),
            clock2ns(send_end - send_start),
            clock2ns(recv_end - recv_start));
        return result;
    }
}

#[no_mangle]
pub extern "C" fn cudaMallocManaged(
    devPtr: MemPtr, // void**
    size: size_t,
    flags: c_uint,
) -> cudaError_t{
    // should update devPtr
    info!(
        "[{}:{}] cudaMallocManaged",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 11;
    let mut result: cudaError_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match size.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send size: {:?}", e),
    }
    match flags.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send flags: {:?}", e),
    }
    channel_sender.flush_out().unwrap();
    let mut devPtr_: MemPtr = Default::default();
    match devPtr_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{*(devPtr as *mut MemPtr) = devPtr_};
        }
        Err(e) => panic!("failed to send devPtr: {:?}", e),
    }
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}


pub extern "C" fn cudaHostAlloc(
    pHost: *mut *mut ::std::os::raw::c_void,
    size: size_t,
    flags: c_uint,
) -> cudaError_t {
    info!(
        "[{}:{}] cudaHostAlloc",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 13;
    let mut result: cudaError_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match size.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send size: {:?}", e),
    }
    match flags.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send flags: {:?}", e),
    }
    channel_sender.flush_out().unwrap();
    let mut pHost_: MemPtr = Default::default();
    match pHost_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{*(pHost as *mut MemPtr) = pHost_};
        }
        Err(e) => panic!("failed to send pHost: {:?}", e),
    }
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}