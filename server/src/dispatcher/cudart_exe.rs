#![allow(non_snake_case)]
use super::*;
use std::alloc::{alloc, dealloc, Layout};

gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_exe!("cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_exe!(
    "cudaGetDeviceCount",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
gen_exe!("cudaGetLastError", "cudaError_t");
gen_exe!("cudaPeekAtLastError", "cudaError_t");
gen_exe!("cudaStreamSynchronize", "cudaError_t", "cudaStream_t");
gen_exe!("cudaMalloc", "cudaError_t", "*mut MemPtr", "usize");

// gen_exe!(
//     "cudaMemcpy",
//     "cudaError_t",
//     "MemPtr",
//     "MemPtr",
//     "usize",
//     "cudaMemcpyKind"
// );
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

    let result = unsafe { cudaMemcpy(dst, src, count, kind) };

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        let data = unsafe { std::slice::from_raw_parts(data_buf as *const u8, count) };
        data.send(channel_sender).unwrap();
        unsafe { dealloc(data_buf, Layout::from_size_align(count, 1).unwrap()) };
    }
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

gen_exe!(
    "cudaStreamIsCapturing",
    "cudaError_t",
    "cudaStream_t",
    "*mut cudaStreamCaptureStatus"
);

gen_exe!(
    "cudaGetDeviceProperties",
    "cudaError_t",
    "*mut cudaDeviceProp",
    "::std::os::raw::c_int"
);
