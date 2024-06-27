#![allow(non_snake_case)]

use std::ffi::c_float;

use super::*;
use cudasys::cublas::*;
use std::os::raw::*;

pub fn cublasCreate_v2Exe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cublasCreate_v2", std::file!(), std::line!());
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
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
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    let result = unsafe {
        cublasSgemm_v2(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            &alpha_,
            A as *const f32,
            lda,
            B as *const f32,
            ldb,
            &beta_,
            C as *mut f32,
            ldc,
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        if let Err(e) = result.send(channel_sender) {
            error!("Error sending result: {:?}", e);
        }
        channel_sender.flush_out().unwrap();
    }
}

pub fn cublasSgemmStridedBatchedExe<T: CommChannel>(
    channel_sender: &mut T,
    channel_receiver: &mut T,
) {
    info!(
        "[{}:{}] cublasSgemmStridedBatched",
        std::file!(),
        std::line!()
    );
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
    if let Err(e) = strideA.recv(channel_receiver) {
        error!("Error receiving strideA: {:?}", e);
    }
    if let Err(e) = B.recv(channel_receiver) {
        error!("Error receiving B: {:?}", e);
    }
    if let Err(e) = ldb.recv(channel_receiver) {
        error!("Error receiving ldb: {:?}", e);
    }
    if let Err(e) = strideB.recv(channel_receiver) {
        error!("Error receiving strideB: {:?}", e);
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
    if let Err(e) = strideC.recv(channel_receiver) {
        error!("Error receiving strideC: {:?}", e);
    }
    if let Err(e) = batchCount.recv(channel_receiver) {
        error!("Error receiving batchCount: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }

    let result = unsafe {
        cublasSgemmStridedBatched(
            handle, transa, transb, m, n, k, 
            &alpha_, A as *const f32, lda, strideA, B as *const f32, ldb, strideB, 
            &beta_, C as *mut f32, ldc, strideC, batchCount
        ) 
    };
    #[cfg(not(feature = "async_api"))]
    {
        result.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}

pub fn cublasGemmExExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cubalsGemmEx", std::file!(), std::line!());
    let mut handle: cublasHandle_t = Default::default();
    let mut transa: cublasOperation_t = Default::default();
    let mut transb: cublasOperation_t = Default::default();
    let mut m: c_int = Default::default();
    let mut n: c_int = Default::default();
    let mut k: c_int = Default::default();
    let mut alpha: c_float = Default::default();
    let mut A: MemPtr = Default::default();
    let mut Atype: cudaDataType = Default::default();
    let mut lda: c_int = Default::default();
    let mut B: MemPtr = Default::default();
    let mut Btype: cudaDataType = Default::default();
    let mut ldb: c_int = Default::default();
    let mut beta: c_float = Default::default();
    let mut C: MemPtr = Default::default();
    let mut Ctype: cudaDataType = Default::default();
    let mut ldc: c_int = Default::default();
    let mut computeType: cublasComputeType_t = Default::default();
    let mut algo: cublasGemmAlgo_t = Default::default();
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
    if let Err(e) = alpha.recv(channel_receiver) {
        error!("Error receiving alpha: {:?}", e);
    }
    if let Err(e) = A.recv(channel_receiver) {
        error!("Error receiving A: {:?}", e);
    }
    if let Err(e) = Atype.recv(channel_receiver) {
        error!("Error receiving Atype: {:?}", e);
    }
    if let Err(e) = lda.recv(channel_receiver) {
        error!("Error receiving lda: {:?}", e);
    }
    if let Err(e) = B.recv(channel_receiver) {
        error!("Error receiving B: {:?}", e);
    }
    if let Err(e) = Btype.recv(channel_receiver) {
        error!("Error receiving Btype: {:?}", e);
    }
    if let Err(e) = ldb.recv(channel_receiver) {
        error!("Error receiving ldb: {:?}", e);
    }
    if let Err(e) = beta.recv(channel_receiver) {
        error!("Error receiving beta: {:?}", e);
    }
    if let Err(e) = C.recv(channel_receiver) {
        error!("Error receiving C: {:?}", e);
    } 
    if let Err(e) = Ctype.recv(channel_receiver) {
        error!("Error receiving Ctype: {:?}", e);
    }
    if let Err(e) = ldc.recv(channel_receiver) {
        error!("Error receiving ldc: {:?}", e);
    }
    if let Err(e) = computeType.recv(channel_receiver) {
        error!("Error receiving computeType: {:?}", e);
    }
    if let Err(e) = algo.recv(channel_receiver) {
        error!("Error receiving algo: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    // convert c_float to *const c_void
    let alpha = &alpha as *const c_float;
    let beta = &beta as *const c_float;
    let result = unsafe {
        cublasGemmEx(handle, transa, transb, m, n, k, 
            alpha as *const c_void, A as *const c_void, Atype, lda,
            B as *const c_void, Btype, ldb, beta as *const c_void,
            C as *mut c_void, Ctype, ldc, computeType, algo)
    };
    #[cfg(not(feature = "async_api"))]
    {
        result.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}

pub fn cublasGemmStridedBatchedExExe<T: CommChannel>(channel_sender: &mut T, channel_receiver: &mut T) {
    info!("[{}:{}] cublasGemmStridedBatchedEx", std::file!(), std::line!());
    let mut handle: cublasHandle_t = Default::default();
    let mut transa: cublasOperation_t = Default::default();
    let mut transb: cublasOperation_t = Default::default();
    let mut m: c_int = Default::default();
    let mut n: c_int = Default::default();
    let mut k: c_int = Default::default();
    let mut alpha: c_float = Default::default();
    let mut A: MemPtr = Default::default();
    let mut Atype: cudaDataType = Default::default();
    let mut lda: c_int = Default::default();
    let mut strideA: c_longlong = Default::default();
    let mut B: MemPtr = Default::default();
    let mut Btype: cudaDataType = Default::default();
    let mut ldb: c_int = Default::default();
    let mut strideB: c_longlong = Default::default();
    let mut beta: c_float = Default::default();
    let mut C: MemPtr = Default::default();
    let mut Ctype: cudaDataType = Default::default();
    let mut ldc: c_int = Default::default();
    let mut strideC: c_longlong = Default::default();
    let mut batchCount: c_int = Default::default();
    let mut computeType: cublasComputeType_t = Default::default();
    let mut algo: cublasGemmAlgo_t = Default::default();
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
    if let Err(e) = alpha.recv(channel_receiver) {
        error!("Error receiving alpha: {:?}", e);
    }
    if let Err(e) = A.recv(channel_receiver) {
        error!("Error receiving A: {:?}", e);
    }
    if let Err(e) = Atype.recv(channel_receiver) {
        error!("Error receiving Atype: {:?}", e);
    }
    if let Err(e) = lda.recv(channel_receiver) {
        error!("Error receiving lda: {:?}", e);
    }
    if let Err(e) = strideA.recv(channel_receiver) {
        error!("Error receiving strideA: {:?}", e);
    }
    if let Err(e) = B.recv(channel_receiver) {
        error!("Error receiving B: {:?}", e);
    }
    if let Err(e) = Btype.recv(channel_receiver) {
        error!("Error receiving Btype: {:?}", e);
    }
    if let Err(e) = ldb.recv(channel_receiver) {
        error!("Error receiving ldb: {:?}", e);
    }
    if let Err(e) = strideB.recv(channel_receiver) {
        error!("Error receiving strideB: {:?}", e);
    }
    if let Err(e) = beta.recv(channel_receiver) {
        error!("Error receiving beta: {:?}", e);
    }
    if let Err(e) = C.recv(channel_receiver) {
        error!("Error receiving C: {:?}", e);
    }
    if let Err(e) = Ctype.recv(channel_receiver) {
        error!("Error receiving Ctype: {:?}", e);
    }
    if let Err(e) = ldc.recv(channel_receiver) {
        error!("Error receiving ldc: {:?}", e);
    }
    if let Err(e) = strideC.recv(channel_receiver) {
        error!("Error receiving strideC: {:?}", e);
    }
    if let Err(e) = batchCount.recv(channel_receiver) {
        error!("Error receiving batchCount: {:?}", e);
    }
    if let Err(e) = computeType.recv(channel_receiver) {
        error!("Error receiving computeType: {:?}", e);
    }
    if let Err(e) = algo.recv(channel_receiver) {
        error!("Error receiving algo: {:?}", e);
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    // convert c_float to *const c_void
    let alpha = &alpha as *const c_float;
    let beta = &beta as *const c_float;
    let result = unsafe {
        cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, 
            alpha as *const c_void, A as *const c_void, Atype, lda, strideA,
            B as *const c_void, Btype, ldb, strideB, beta as *const c_void,
            C as *mut c_void, Ctype, ldc, strideC, batchCount, computeType, algo
        )
    };
    #[cfg(not(feature = "async_api"))]
    {
        result.send(channel_sender).unwrap();
        channel_sender.flush_out().unwrap();
    }
}
