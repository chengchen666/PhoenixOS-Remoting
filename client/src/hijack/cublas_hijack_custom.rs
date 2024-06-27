
#![allow(non_snake_case)]

use super::*;
use cudasys::types::cublas::*;
use ::std::os::raw::*;


#[no_mangle]
pub extern "C" fn cublasCreate_v2(
    handle: *mut cublasHandle_t,
) -> cublasStatus_t {
    info!("[{}:{}] cublasCreate_v2", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 2000;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    // receive handle from server
    let mut handle_: cublasHandle_t = Default::default();
    let mut result: cublasStatus_t = Default::default(); 
    match handle_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{ *handle = handle_ };
        }
        Err(e) => error!("Error receiving handle: {:?}", e),
    }
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => error!("Error receiving result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t, 
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,          // assume alpha is on host
    A: MemPtr,
    lda: c_int,
    B: MemPtr,
    ldb: c_int,
    beta: *const c_float,           // assume beta is on host
    C: MemPtr,
    ldc: c_int,
) -> cublasStatus_t {
    info!("[{}:{}] cublasSgemm_v2", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 2004;
    let alpha_ = unsafe { *alpha };
    let beta_ = unsafe { *beta };
    let mut result: cublasStatus_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    if let Err(e) = handle.send(channel_sender) {
        error!("Error sending handle: {:?}", e)
    }
    if let Err(e) = transa.send(channel_sender) {
        error!("Error sending transa: {:?}", e)
    }
    if let Err(e) = transb.send(channel_sender) {
        error!("Error sending transb: {:?}", e)
    }
    if let Err(e) = m.send(channel_sender) {
        error!("Error sending m: {:?}", e)
    }
    if let Err(e) = n.send(channel_sender) {
        error!("Error sending n: {:?}", e)
    }
    if let Err(e) = k.send(channel_sender) {
        error!("Error sending k: {:?}", e)
    }
    if let Err(e) = alpha_.send(channel_sender) {
        error!("Error sending alpha: {:?}", e)
    }
    if let Err(e) = A.send(channel_sender) {
        error!("Error sending A: {:?}", e)
    }
    if let Err(e) = lda.send(channel_sender) {
        error!("Error sending lda: {:?}", e)
    }
    if let Err(e) = B.send(channel_sender) {
        error!("Error sending B: {:?}", e)
    }
    if let Err(e) = ldb.send(channel_sender) {
        error!("Error sending ldb: {:?}", e)
    }
    if let Err(e) = beta_.send(channel_sender) {
        error!("Error sending beta: {:?}", e)
    }
    if let Err(e) = C.send(channel_sender) {
        error!("Error sending C: {:?}", e)
    }
    if let Err(e) = ldc.send(channel_sender) {
        error!("Error sending ldc: {:?}", e)
    }
    channel_sender.flush_out().unwrap();

    #[cfg(feature = "async_api")]
    {
        return cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    }
    #[cfg(not(feature = "async_api"))]
    {
        if let Err(e) = result.recv(channel_receiver) {
            error!("Error receiving result: {:?}", e)
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        return result;
    }
}

#[no_mangle]
pub extern "C" fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    A: MemPtr,
    lda: c_int,
    strideA: c_longlong,
    B: MemPtr,
    ldb: c_int,
    strideB: c_longlong,
    beta: *const c_float,
    C: MemPtr,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> cublasStatus_t {
    info!("[{}:{}] cublasSgemmStridedBatched", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 2005;
    let alpha_ = unsafe { *alpha };
    let beta_ = unsafe { *beta };
    let mut result: cublasStatus_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    if let Err(e) = handle.send(channel_sender) {
        error!("Error sending handle: {:?}", e)
    }
    if let Err(e) = transa.send(channel_sender) {
        error!("Error sending transa: {:?}", e)
    }
    if let Err(e) = transb.send(channel_sender) {
        error!("Error sending transb: {:?}", e)
    }
    if let Err(e) = m.send(channel_sender) {
        error!("Error sending m: {:?}", e)
    }
    if let Err(e) = n.send(channel_sender) {
        error!("Error sending n: {:?}", e)
    }
    if let Err(e) = k.send(channel_sender) {
        error!("Error sending k: {:?}", e)
    }
    if let Err(e) = alpha_.send(channel_sender) {
        error!("Error sending alpha: {:?}", e)
    }
    if let Err(e) = A.send(channel_sender) {
        error!("Error sending A: {:?}", e)
    }
    if let Err(e) = lda.send(channel_sender) {
        error!("Error sending lda: {:?}", e)
    }
    if let Err(e) = strideA.send(channel_sender) {
        error!("Error sending strideA: {:?}", e)
    }
    if let Err(e) = B.send(channel_sender) {
        error!("Error sending B: {:?}", e)
    }
    if let Err(e) = ldb.send(channel_sender) {
        error!("Error sending ldb: {:?}", e)
    }
    if let Err(e) = strideB.send(channel_sender) {
        error!("Error sending strideB: {:?}", e)
    }
    if let Err(e) = beta_.send(channel_sender) {
        error!("Error sending beta: {:?}", e)
    }
    if let Err(e) = C.send(channel_sender) {
        error!("Error sending C: {:?}", e)
    }
    if let Err(e) = ldc.send(channel_sender) {
        error!("Error sending ldc: {:?}", e)
    }
    if let Err(e) = strideC.send(channel_sender) {
        error!("Error sending strideC: {:?}", e)
    }
    if let Err(e) = batchCount.send(channel_sender) {
        error!("Error sending batchCount: {:?}", e)
    }

    #[cfg(feature = "async_api")]
    return cublasStatus_t::CUBLAS_STATUS_SUCCESS;

    channel_sender.flush_out().unwrap();
    if let Err(e) = result.recv(channel_receiver) {
        error!("Error receiving result: {:?}", e)
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    A: MemPtr,
    Atype: cudaDataType,
    lda: c_int,
    B: MemPtr,
    Btype: cudaDataType,
    ldb: c_int,
    beta: *const c_float,
    C: MemPtr,
    Ctype: cudaDataType,
    ldc: c_int,
    computeType: cudaDataType,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    info!("[{}:{}] cublasGemmEx", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 2007;
    let alpha_ = unsafe { *alpha };
    let beta_ = unsafe { *beta };
    let mut result: cublasStatus_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    if let Err(e) = handle.send(channel_sender) {
        error!("Error sending handle: {:?}", e)
    }
    if let Err(e) = transa.send(channel_sender) {
        error!("Error sending transa: {:?}", e)
    }
    if let Err(e) = transb.send(channel_sender) {
        error!("Error sending transb: {:?}", e)
    }
    if let Err(e) = m.send(channel_sender) {
        error!("Error sending m: {:?}", e)
    }
    if let Err(e) = n.send(channel_sender) {
        error!("Error sending n: {:?}", e)
    }
    if let Err(e) = k.send(channel_sender) {
        error!("Error sending k: {:?}", e)
    }
    if let Err(e) = alpha_.send(channel_sender) {
        error!("Error sending alpha: {:?}", e)
    }
    if let Err(e) = A.send(channel_sender) {
        error!("Error sending A: {:?}", e)
    }
    if let Err(e) = Atype.send(channel_sender) {
        error!("Error sending Atype: {:?}", e)
    }
    if let Err(e) = lda.send(channel_sender) {
        error!("Error sending lda: {:?}", e)
    }
    if let Err(e) = B.send(channel_sender) {
        error!("Error sending B: {:?}", e)
    }
    if let Err(e) = Btype.send(channel_sender) {
        error!("Error sending Btype: {:?}", e)
    }
    if let Err(e) = ldb.send(channel_sender) {
        error!("Error sending ldb: {:?}", e)
    }
    if let Err(e) = beta_.send(channel_sender) {
        error!("Error sending beta: {:?}", e)
    }
    if let Err(e) = C.send(channel_sender) {
        error!("Error sending C: {:?}", e)
    }
    if let Err(e) = Ctype.send(channel_sender) {
        error!("Error sending Ctype: {:?}", e)
    }
    if let Err(e) = ldc.send(channel_sender) {
        error!("Error sending ldc: {:?}", e)
    }
    if let Err(e) = computeType.send(channel_sender) {
        error!("Error sending computeType: {:?}", e)
    }
    if let Err(e) = algo.send(channel_sender) {
        error!("Error sending algo: {:?}", e)
    }
    channel_sender.flush_out().unwrap();
    #[cfg(feature = "async_api")]
    return cublasStatus_t::CUBLAS_STATUS_SUCCESS;

    #[cfg(not(feature = "async_api"))]
    {
        if let Err(e) = result.recv(channel_receiver) {
            error!("Error receiving result: {:?}", e)
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        return result;
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    result
}

#[no_mangle]
pub extern "C" fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    A: MemPtr,
    Atype: cudaDataType,
    lda: c_int,
    strideA: c_longlong,
    B: MemPtr,
    Btype: cudaDataType,
    ldb: c_int,
    strideB: c_longlong,
    beta: *const c_float,
    C: MemPtr,
    Ctype: cudaDataType,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
    computeType: cudaDataType,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    info!("[{}:{}] cublasGemmStridedBatchedEx", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 2008;
    let alpha_ = unsafe { *alpha };
    let beta_ = unsafe { *beta };
    let mut result: cublasStatus_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    if let Err(e) = handle.send(channel_sender) {
        error!("Error sending handle: {:?}", e)
    }
    if let Err(e) = transa.send(channel_sender) {
        error!("Error sending transa: {:?}", e)
    }
    if let Err(e) = transb.send(channel_sender) {
        error!("Error sending transb: {:?}", e)
    }
    if let Err(e) = m.send(channel_sender) {
        error!("Error sending m: {:?}", e)
    }
    if let Err(e) = n.send(channel_sender) {
        error!("Error sending n: {:?}", e)
    }
    if let Err(e) = k.send(channel_sender) {
        error!("Error sending k: {:?}", e)
    }
    if let Err(e) = alpha_.send(channel_sender) {
        error!("Error sending alpha: {:?}", e)
    }
    if let Err(e) = A.send(channel_sender) {
        error!("Error sending A: {:?}", e)
    }
    if let Err(e) = Atype.send(channel_sender) {
        error!("Error sending Atype: {:?}", e)
    }
    if let Err(e) = lda.send(channel_sender) {
        error!("Error sending lda: {:?}", e)
    }
    if let Err(e) = strideA.send(channel_sender) {
        error!("Error sending strideA: {:?}", e)
    }
    if let Err(e) = B.send(channel_sender) {
        error!("Error sending B: {:?}", e)
    }
    if let Err(e) = Btype.send(channel_sender) {
        error!("Error sending Btype: {:?}", e)
    }
    if let Err(e) = ldb.send(channel_sender) {
        error!("Error sending ldb: {:?}", e)
    }
    if let Err(e) = strideB.send(channel_sender) {
        error!("Error sending strideB: {:?}", e)
    }
    if let Err(e) = beta_.send(channel_sender) {
        error!("Error sending beta: {:?}", e)
    }
    if let Err(e) = C.send(channel_sender) {
        error!("Error sending C: {:?}", e)
    }
    if let Err(e) = Ctype.send(channel_sender) {
        error!("Error sending Ctype: {:?}", e)
    }
    if let Err(e) = ldc.send(channel_sender) {
        error!("Error sending ldc: {:?}", e)
    }
    if let Err(e) = strideC.send(channel_sender) {
        error!("Error sending strideC: {:?}", e)
    }
    if let Err(e) = batchCount.send(channel_sender) {
        error!("Error sending batchCount: {:?}", e)
    }
    if let Err(e) = computeType.send(channel_sender) {
        error!("Error sending computeType: {:?}", e)
    }
    if let Err(e) = algo.send(channel_sender) {
        error!("Error sending algo: {:?}", e)
    }
    channel_sender.flush_out().unwrap();

    #[cfg(feature = "async_api")]
    return cublasStatus_t::CUBLAS_STATUS_SUCCESS;

    if let Err(e) = result.recv(channel_receiver) {
        error!("Error receiving result: {:?}", e)
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    result
}
