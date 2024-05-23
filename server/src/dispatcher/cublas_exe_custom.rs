#![allow(non_snake_case)]

use std::ffi::c_float;

use super::*;
use cudasys::cublas::*;
use ::std::os::raw::*;

pub fn cublasCreate_v2Exe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cublasCreate_v2", std::file!(), std::line!());
    let mut handle: cublasHandle_t = Default::default();
    let result = unsafe { cublasCreate_v2(&mut handle) };

    handle.send(channel_sender).unwrap();
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();
}

pub fn cublasSgemm_v2Exe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cublasSgemm_v2Exe", std::file!(), std::line!());
    let mut handle: cublasHandle_t = Default::default();
    let mut transa: cublasOperation_t = Default::default();
    let mut transb: cublasOperation_t = Default::default();
    let mut m: c_int = Default::default();
    let mut n: c_int = Default::default();
    let mut k: c_int = Default::default();
    let mut alpha_: c_float = Default::default();
    let mut A: MemPtr = Default::default();
    let mut lda: c_int = Default::default();
    let mut B: MemPtr = Default::default();
    let mut ldb: c_int = Default::default();
    let mut beta_: c_float = Default::default();
    let mut C: MemPtr = Default::default();
    let mut ldc: c_int = Default::default();
    if let Err(e) = handle.recv(channel_receiver) {
        error!("Error receiving handle: {:?}", e);
    }
    if let Err(e) = transa.recv(channel_receiver) {
        error!("Error receiving transa: {:?}", e);
    }
    if let Err(e) = transb.recv(channel_receiver) {
        error!("Error receiving transb: {:?}", e);
    }
    if let Err(e) = m.recv(channel_receiver) {
        error!("Error receiving m: {:?}", e);
    }
    if let Err(e) = n.recv(channel_receiver) {
        error!("Error receiving n: {:?}", e);
    }
    if let Err(e) = k.recv(channel_receiver) {
        error!("Error receiving k: {:?}", e);
    }
    if let Err(e) = alpha_.recv(channel_receiver) {
        error!("Error receiving alpha: {:?}", e);
    }
    if let Err(e) = A.recv(channel_receiver) {
        error!("Error receiving A: {:?}", e);
    }
    if let Err(e) = lda.recv(channel_receiver) {
        error!("Error receiving lda: {:?}", e);
    }
    if let Err(e) = B.recv(channel_receiver) {
        error!("Error receiving B: {:?}", e);
    }
    if let Err(e) = ldb.recv(channel_receiver) {
        error!("Error receiving ldb: {:?}", e);
    }
    if let Err(e) = beta_.recv(channel_receiver) {
        error!("Error receiving beta: {:?}", e);
    }
    if let Err(e) = C.recv(channel_receiver) {
        error!("Error receiving C: {:?}", e);
    }
    if let Err(e) = ldc.recv(channel_receiver) {
        error!("Error receiving ldc: {:?}", e);
    }
    let result = unsafe { 
        cublasSgemm_v2(
            handle, transa, transb, m, n, k,
            &alpha_, A as *const f32, lda, B as *const f32, ldb, 
            &beta_, C as *mut f32, ldc
        ) 
    };
    if let Err(e) = result.send(channel_sender) {
        error!("Error sending result: {:?}", e);
    }
    channel_sender.flush_out().unwrap();
}

pub fn cublasSgemmStridedBatchedExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cublasSgemmStridedBatched", std::file!(), std::line!());
    let mut handle: cublasHandle_t = Default::default();
    let mut transa: cublasOperation_t = Default::default();
    let mut transb: cublasOperation_t = Default::default();
    let mut m: c_int = Default::default();
    let mut n: c_int = Default::default();
    let mut k: c_int = Default::default();
    let mut alpha_: c_float = Default::default();
    let mut A: MemPtr = Default::default();
    let mut lda: c_int = Default::default();
    let mut strideA: c_longlong = Default::default();
    let mut B: MemPtr = Default::default();
    let mut ldb: c_int = Default::default();
    let mut strideB: c_longlong = Default::default();
    let mut beta_: c_float = Default::default();
    let mut C: MemPtr = Default::default();
    let mut ldc: c_int = Default::default();
    let mut strideC: c_longlong = Default::default();
    let mut batchCount: c_int = Default::default();

    handle.recv(channel_receiver).unwrap();
    transa.recv(channel_receiver).unwrap();
    transb.recv(channel_receiver).unwrap();
    m.recv(channel_receiver).unwrap();
    n.recv(channel_receiver).unwrap();
    k.recv(channel_receiver).unwrap();
    alpha_.recv(channel_receiver).unwrap();
    A.recv(channel_receiver).unwrap();
    lda.recv(channel_receiver).unwrap();
    strideA.recv(channel_receiver).unwrap();
    B.recv(channel_receiver).unwrap();
    ldb.recv(channel_receiver).unwrap();
    strideB.recv(channel_receiver).unwrap();
    beta_.recv(channel_receiver).unwrap();
    C.recv(channel_receiver).unwrap();
    ldc.recv(channel_receiver).unwrap();
    strideC.recv(channel_receiver).unwrap();
    batchCount.recv(channel_receiver).unwrap();

    let result = unsafe { 
        cublasSgemmStridedBatched(
            handle, transa, transb, m, n, k, 
            &alpha_, A as *const f32, lda, strideA, B as *const f32, ldb, strideB, 
            &beta_, C as *mut f32, ldc, strideC, batchCount
        ) 
    };
    result.send(channel_sender).unwrap();
    channel_sender.flush_out().unwrap();

}