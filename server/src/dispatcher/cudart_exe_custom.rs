#![allow(non_snake_case)]
use super::*;
use cudasys::cudart::*;
use std::os::raw::*;
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::null_mut,
};

pub fn cudaMemcpyExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaMemcpy", std::file!(), std::line!());
    let mut dst: MemPtr = Default::default();
    dst.recv(channel_receiver).unwrap();
    let mut src: MemPtr = Default::default();
    src.recv(channel_receiver).unwrap();
    let mut count: usize = Default::default();
    count.recv(channel_receiver).unwrap();
    let mut kind: cudaMemcpyKind = Default::default();
    kind.recv(channel_receiver).unwrap();

    let mut data_buf = 0 as *mut u8;

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        data_buf = unsafe { alloc(Layout::from_size_align(count, 1).unwrap()) };
        if data_buf.is_null() {
            panic!("failed to allocate data_buf");
        }
        let data = unsafe { std::slice::from_raw_parts_mut(data_buf, count) };
        data.recv(channel_receiver).unwrap();
        src = data_buf as MemPtr;
    } else if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        data_buf = unsafe { alloc(Layout::from_size_align(count, 1).unwrap()) };
        if data_buf.is_null() {
            panic!("failed to allocate data_buf");
        }
        dst = data_buf as MemPtr;
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let result = unsafe {
        cudaMemcpy(
            dst as *mut std::os::raw::c_void,
            src as *const std::os::raw::c_void,
            count as size_t,
            kind,
        )
    };

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        let data = unsafe { std::slice::from_raw_parts(data_buf as *const u8, count) };
        data.send(channel_sender).unwrap();
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
    #[cfg(not(feature = "async_api"))]
    {
        result.send(channel_sender).unwrap();
    }

    channel_sender.flush_out().unwrap();
}

pub fn cudaMallocExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaMalloc", std::file!(), std::line!());
    let mut param1 = 0 as *mut ::std::os::raw::c_void;
    let mut param2: usize = Default::default();
    param2.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result: cudaError_t = unsafe {
        cudaMalloc(
            &mut param1 as *mut *mut ::std::os::raw::c_void,
            param2 as size_t,
        )
    };
    let param1 = param1 as MemPtr;
    param1.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudaFreeExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaFree", std::file!(), std::line!());
    let mut param1: MemPtr = Default::default();
    param1.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result: cudaError_t = unsafe { cudaFree(param1 as *mut ::std::os::raw::c_void) };

    #[cfg(not(feature = "async_api"))]
    {
        result.send(channel_sender).unwrap();
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudaLaunchKernelExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaLaunchKernel", std::file!(), std::line!());
    let mut func: MemPtr = Default::default();
    func.recv(channel_receiver).unwrap();
    let mut gridDim: dim3 = Default::default();
    gridDim.recv(channel_receiver).unwrap();
    let mut blockDim: dim3 = Default::default();
    blockDim.recv(channel_receiver).unwrap();
    let mut argc: usize = Default::default();
    argc.recv(channel_receiver).unwrap();
    let mut arg_vec: Vec<Vec<u8>> = Vec::new();
    for _ in 0..argc {
        let mut arg: Vec<u8> = Default::default();
        arg.recv(channel_receiver).unwrap();
        arg_vec.push(arg);
    }
    let mut args: Vec<*mut std::os::raw::c_void> = Vec::new();
    for i in 0..argc {
        args.push(arg_vec[i].as_ptr() as *mut std::os::raw::c_void);
    }
    let mut sharedMem: usize = Default::default();
    sharedMem.recv(channel_receiver).unwrap();
    let mut stream: cudaStream_t = Default::default();
    stream.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let device_func = get_function(func).unwrap();
    let result = unsafe {
        cudasys::cuda::cuLaunchKernel(
            device_func,
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            sharedMem as std::os::raw::c_uint,
            stream as cudasys::cuda::CUstream,
            args.as_mut_ptr() as *mut *mut std::os::raw::c_void,
            std::ptr::null_mut(),
        )
    };

    #[cfg(not(feature = "async_api"))]
    {
        result.send(channel_sender).unwrap();
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudaMallocManagedExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaMallocManaged", std::file!(), std::line!());
    let mut size: size_t = Default::default();
    let mut flags: c_uint = Default::default();
    match size.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving size: {:?}", e);
        }
    }
    match flags.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving flags: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let mut devPtr: *mut c_void = null_mut();
    let result = unsafe { cudaMallocManaged(&mut devPtr, size, flags) };
    let devPtr = devPtr as MemPtr;
    match devPtr.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("failed to sending devPtr: {:?}", e);
        }
    }
    match result.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("failed to sending result: {:?}", e);
        }
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudaPointerGetAttributesExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cudaPointerGetAttributes",
        std::file!(),
        std::line!()
    );
    let mut attributes: cudaPointerAttributes = Default::default();
    let mut ptr: MemPtr = Default::default();
    ptr.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = unsafe {
        cudaPointerGetAttributes(
            &mut attributes as *mut cudaPointerAttributes,
            ptr as *const std::os::raw::c_void,
        )
    };
    attributes.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cudaHostAllocExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaHostAlloc", std::file!(), std::line!());
    let mut pHost: *mut ::std::os::raw::c_void = null_mut();
    let mut size: size_t = Default::default();
    let mut flags: c_uint = Default::default();
    if let Err(e) = size.recv(channel_receiver) {
        error!("failed to receive size: {:?}", e);
    }
    if let Err(e) = flags.recv(channel_receiver) {
        error!("failed to receive flags: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = unsafe { cudaHostAlloc(&mut pHost, size, flags) };
    // send back pHost as MemPtr
    let pHost_ = pHost as MemPtr;
    if let Err(e) = pHost_.send(channel_sender) {
        error!("failed to send pHost: {:?}", e);
    }
    if let Err(e) = result.send(channel_sender) {
        error!("failed to send result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}

pub fn cudaFuncGetAttributesExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cudaFuncGetAttributes", std::file!(), std::line!());
    let mut attributes: cudaFuncAttributes = Default::default();
    let mut func: MemPtr = Default::default();
    func.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = unsafe {
        cudaFuncGetAttributes(
            &mut attributes as *mut cudaFuncAttributes,
            func as *const std::os::raw::c_void,
        )
    };
    attributes.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}
