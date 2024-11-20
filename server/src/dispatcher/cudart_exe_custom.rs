#![cfg_attr(not(feature = "phos"), expect(unused_variables))]

use super::*;
use cudasys::cudart::*;
use std::alloc::{alloc, dealloc, Layout};

pub fn cudaMemcpyExe<C: CommChannel>(proc_id: i32, server: &mut ServerWorker<C>) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
    log::debug!("[{}:{}] cudaMemcpy", std::file!(), std::line!());

    let mut dst: MemPtr = Default::default();
    dst.recv(channel_receiver).unwrap();
    let mut src: MemPtr = Default::default();
    src.recv(channel_receiver).unwrap();
    let mut count: usize = Default::default();
    count.recv(channel_receiver).unwrap();
    let mut kind: cudaMemcpyKind = cudaMemcpyKind::cudaMemcpyHostToHost;
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

    #[cfg(not(feature = "phos"))]
    let result = unsafe {
        cudaMemcpy(
            dst as *mut std::os::raw::c_void,
            src as *const std::os::raw::c_void,
            count as usize,
            kind,
        )
    };
    #[cfg(feature = "phos")]
    let result = cudaError_t::from_i32(
        match kind {
            cudaMemcpyKind::cudaMemcpyHostToDevice => pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                440,
                0u64,
                &[
                    addr_of(&dst), size_of_val(&dst),
                    src as usize, count as usize,
                ],
                0u64,
                0u64,
            ),
            cudaMemcpyKind::cudaMemcpyDeviceToHost => pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                441,
                0u64,
                &[
                    addr_of(&src), size_of_val(&src),
                    addr_of(&count), size_of_val(&count),
                ],
                dst as u64,
                count as u64,
            ),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice => pos_process(
                POS_CUDA_WS.lock().unwrap().get_ptr(),
                443,
                0u64,
                &[
                    addr_of(&dst), size_of_val(&dst),
                    addr_of(&src), size_of_val(&src),
                    addr_of(&count), size_of_val(&count),
                ],
                0u64,
                0u64,
            ),
            _ => panic!("Illegal cudaMemcpy kind"),
        }
    ).expect("Illegal result ID");


    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        let data = unsafe { std::slice::from_raw_parts(data_buf as *const u8, count) };
        data.send(channel_sender).unwrap();
        if cfg!(feature = "async_api") {
            channel_sender.flush_out().unwrap();
        }
    }
    if cfg!(not(feature = "async_api")) {
        result.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
}

pub fn cudaGetErrorStringExe<C: CommChannel>(proc_id: i32, server: &mut ServerWorker<C>) {
    let ServerWorker { channel_sender, channel_receiver, .. } = server;
    log::debug!("[{}:{}] cudaGetErrorString", std::file!(), std::line!());
    let mut error: cudaError_t = Default::default();
    error.recv(channel_receiver).unwrap();
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    #[cfg(not(feature = "phos"))]
    let result = unsafe { cudaGetErrorString(error) };
    #[cfg(feature = "phos")]
    let result = pos_process(
        POS_CUDA_WS.lock().unwrap().get_ptr(),
        proc_id,
        0u64,
        &[
            addr_of(&error), size_of_val(&error),
        ],
        0u64,
        0u64,
    ) as *const i8;
    let result = unsafe { std::ffi::CStr::from_ptr(result).to_bytes().to_vec() };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}
