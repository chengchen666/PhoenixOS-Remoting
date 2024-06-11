#![allow(non_snake_case)]

use super::*;
use cudasys::types::cudnn::*;
use ::std::os::raw::*;

#[no_mangle]
pub extern "C" fn cudnnCreate(
    handle: *mut cudnnHandle_t,
) -> cudnnStatus_t{
    info!("[{}:{}] cudnnCreate", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1500;
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
    let mut result: cudnnStatus_t = Default::default();
    let mut handle_: cudnnHandle_t = Default::default();
    // receive handle from server
    match handle_.recv(channel_receiver){
        Ok(()) => {
            unsafe{*handle = handle_};
        }
        Err(e) => {
            error!("failed to receiving handle: {:?}", e);
        }
    }
    match result.recv(channel_receiver){
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if cudnnStatus_t::CUDNN_STATUS_SUCCESS != result {
        panic!("error cudnnCreate: {:?}", result);
    }
    result
}

#[cfg(feature = "shadow_desc")]
#[no_mangle]
pub extern "C" fn cudnnCreateTensorDescriptor(
    tensorDesc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t {
    info!("[{}:{}] cudnnCreateTensorDescriptor", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1501;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let resource_idx: cudnnTensorDescriptor_t = *RESOURCE_IDX.lock().unwrap();
    unsafe { *tensorDesc = resource_idx; }
    *RESOURCE_IDX.lock().unwrap() += 1;
    match resource_idx.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending tensor_desc_addr: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}
#[cfg(not(feature = "shadow_desc"))]
#[no_mangle]
pub extern "C" fn cudnnCreateTensorDescriptor(
    tensorDesc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t{
    info!("[{}:{}] cudnnCreateTensorDescriptor", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1501;
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
    let mut result: cudnnStatus_t = Default::default();
    let mut tensorDesc_: cudnnTensorDescriptor_t = Default::default();
    // receive tensorDesc from server
    match tensorDesc_.recv(channel_receiver){
        Ok(()) => {
            unsafe{*tensorDesc = tensorDesc_};
        }
        Err(e) => {
            error!("failed to receiving tensorDesc: {:?}", e);
        }
    }
    match result.recv(channel_receiver){
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if cudnnStatus_t::CUDNN_STATUS_SUCCESS != result {
        panic!("error cudnnCreateTensorDescriptor: {:?}", result);
    }
    result
}

#[no_mangle]
pub extern "C" fn cudnnCreateActivationDescriptor(
    activationDesc: *mut cudnnPoolingDescriptor_t
) -> cudnnStatus_t{
    info!(
        "[{}:{}] cudnnCreateActivationDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1503;
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
    let mut result: cudnnStatus_t = Default::default();
    let mut activationDesc_: cudnnPoolingDescriptor_t = Default::default();
    // receive activationDesc from server
    match activationDesc_.recv(channel_receiver){
        Ok(()) => {
            unsafe{*activationDesc = activationDesc_};
        }
        Err(e) => {
            error!("failed to receiving activationDesc: {:?}", e);
        }
    }
    match result.recv(channel_receiver){
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if cudnnStatus_t::CUDNN_STATUS_SUCCESS != result {
        panic!("error cudnnCreateActivationDescriptor: {:?}", result);
    }
    result
}


#[no_mangle]
pub extern "C" fn cudnnActivationForward(
    handle: cudnnStatus_t,
    activationDesc: cudnnActivationDescriptor_t,
    alpha: MemPtr,
    xDesc: cudnnTensorDescriptor_t,
    x: MemPtr,
    beta: MemPtr,
    yDesc: cudnnTensorDescriptor_t,
    y: MemPtr, // *mut c_void
) -> cudnnStatus_t{
    info!(
        "[{}:{}] cudnnActivationForward",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1505;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let mut result: cudnnStatus_t = Default::default();
    match handle.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending handle: {:?}", e);
        }
    }
    match activationDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending activationDesc: {:?}", e);
        }
    }
    match alpha.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending alpha: {:?}", e);
        }
    }
    match xDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending xDesc: {:?}", e);
        }
    }
    match x.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending x: {:?}", e);
        }
    }
    match beta.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending beta: {:?}", e);
        }
    }
    match yDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending yDesc: {:?}", e);
        }
    }
    match y.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending y: {:?}", e);
        }
    } 
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    // since y has been created, we don't need to receive the address
    match result.recv(channel_receiver) {
        Ok(_) => {}
        Err(e) => {
            error!("Error receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnSetTensorNdDescriptor(
    tensorDesc:cudnnTensorDescriptor_t ,
    dataType: cudnnDataType_t,
    nbDims:c_int,
    dimA: *const c_int,                       // dimA stored on cpu
    strideA: *const c_int// strideA stored on cpu
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnSetTensorNdDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1509;
    // process dimA and strideA
    let dimA_ = unsafe {
        let slice = std::slice::from_raw_parts(dimA, nbDims as usize);
        slice.to_vec()
    };
    let strideA_ = unsafe {
        let slice = std::slice::from_raw_parts(strideA, nbDims as usize);
        slice.to_vec()
    };
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let mut result: cudnnStatus_t = Default::default();
    match tensorDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending tensorDesc: {:?}", e);
        }
    }
    match dataType.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending dataType: {:?}", e);
        }
    }
    match nbDims.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending nbDims: {:?}", e);
        }
    }
    match dimA_.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending dimA: {:?}", e);
        }
    }
    match strideA_.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending strideA: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    #[cfg(feature = "async_api")]
    {
        return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    }
    #[cfg(not(feature = "async_api"))]
    {
        match result.recv(channel_receiver) {
            Ok(_) => {}
            Err(e) => {
                error!("Error receiving result: {:?}", e);
            }
        }
        match channel_receiver.recv_ts() {
                    Ok(()) => {}
                    Err(e) => panic!("failed to receive timestamp: {:?}", e),
                }
        return result;
    }
}


#[cfg(feature = "shadow_desc")]
#[no_mangle]
pub extern "C" fn cudnnCreateFilterDescriptor(
    filterDesc: *mut cudnnFilterDescriptor_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnCreateFilterDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1511;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let resource_idx: cudnnTensorDescriptor_t = *RESOURCE_IDX.lock().unwrap();
    unsafe { *filterDesc = resource_idx; }
    *RESOURCE_IDX.lock().unwrap() += 1;
    match resource_idx.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending tensor_desc_addr: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}
#[cfg(not(feature = "shadow_desc"))]
#[no_mangle]
pub extern "C" fn cudnnCreateFilterDescriptor(
    filterDesc: *mut cudnnFilterDescriptor_t
) -> cudnnStatus_t{
    info!(
        "[{}:{}] cudnnCreateFilterDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1511;
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
    let mut result: cudnnStatus_t = Default::default();
    let mut filterDesc_: cudnnFilterDescriptor_t = Default::default();
    // receive filterDesc from server
    match filterDesc_.recv(channel_receiver){
        Ok(()) => {
            unsafe{*filterDesc = filterDesc_};
        }
        Err(e) => {
            error!("failed to receiving filterDesc: {:?}", e);
        }
    }
    match result.recv(channel_receiver){
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if cudnnStatus_t::CUDNN_STATUS_SUCCESS != result {
        panic!("error cudnnCreateFilterDescriptor: {:?}", result);
    }
    result
}

#[no_mangle]
pub extern "C" fn cudnnSetFilterNdDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nbDims:c_int,
    filterDimA: *const c_int
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnSetFilterNdDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    // process filterDimA
    let filterDimA_ = unsafe {
        let slice = std::slice::from_raw_parts(filterDimA, nbDims as usize);
        slice.to_vec()
    };
    let proc_id = 1513;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let mut result: cudnnStatus_t = Default::default();
    match filterDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending filterDesc: {:?}", e);
        }
    }
    match dataType.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending dataType: {:?}", e);
        }
    }
    match format.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending format: {:?}", e);
        }
    }
    match nbDims.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending nbDims: {:?}", e);
        }
    }
    match filterDimA_.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending filterDimA: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    #[cfg(feature = "async_api")]
    {
        return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    }
    #[cfg(not(feature = "async_api"))]
    {
        match result.recv(channel_receiver) {
            Ok(_) => {}
            Err(e) => {
                error!("Error receiving result: {:?}", e);
            }
        }
        match channel_receiver.recv_ts() {
                    Ok(()) => {}
                    Err(e) => panic!("failed to receive timestamp: {:?}", e),
                }
        return result;
    }
}


#[cfg(feature = "shadow_desc")]
#[no_mangle]
pub extern "C" fn cudnnCreateConvolutionDescriptor(
    convDesc: *mut cudnnConvolutionDescriptor_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnCreateConvolutionDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1514;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let resource_idx: cudnnTensorDescriptor_t = *RESOURCE_IDX.lock().unwrap();
    unsafe { *convDesc = resource_idx; }
    *RESOURCE_IDX.lock().unwrap() += 1;
    match resource_idx.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending tensor_desc_addr: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    cudnnStatus_t::CUDNN_STATUS_SUCCESS
}
#[cfg(not(feature = "shadow_desc"))]
#[no_mangle]
pub extern "C" fn cudnnCreateConvolutionDescriptor(
    convDesc: *mut cudnnConvolutionDescriptor_t
) -> cudnnStatus_t{
    info!(
        "[{}:{}] cudnnCreateConvolutionDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1514;
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
    let mut result: cudnnStatus_t = Default::default();
    let mut convDesc_: cudnnConvolutionDescriptor_t = Default::default();
    // receive convDesc from server
    match convDesc_.recv(channel_receiver){
        Ok(()) => {
            unsafe{*convDesc = convDesc_};
        }
        Err(e) => {
            error!("failed to receiving convDesc: {:?}", e);
        }
    }
    match result.recv(channel_receiver){
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if cudnnStatus_t::CUDNN_STATUS_SUCCESS != result {
        panic!("error cudnnCreateConvolutionDescriptor: {:?}", result);
    }
    result
}

#[no_mangle]
pub extern "C" fn cudnnSetConvolutionNdDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLength: c_int,
    padA: *const c_int,
    filterStrideA: *const c_int,
    dilationA: *const c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnSetConvolutionNdDescriptor",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    // process padA, filterStrideA, dilationA
    let padA_ = unsafe {
        let slice = std::slice::from_raw_parts(padA, arrayLength as usize);
        slice.to_vec()
    };
    let filterStrideA_ = unsafe {
        let slice = std::slice::from_raw_parts(filterStrideA, arrayLength as usize);
        slice.to_vec()
    };
    let dilationA_ = unsafe {
        let slice = std::slice::from_raw_parts(dilationA, arrayLength as usize);
        slice.to_vec()
    };
    let proc_id = 1516;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let mut result: cudnnStatus_t = Default::default();
    match convDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending convDesc: {:?}", e);
        }
    }
    match arrayLength.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending arrayLength: {:?}", e);
        }
    }
    match padA_.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending padA: {:?}", e);
        }
    }
    match filterStrideA_.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending filterStrideA: {:?}", e);
        }
    }
    match dilationA_.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending dilationA: {:?}", e);
        }
    }
    match mode.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending mode: {:?}", e);
        }
    }
    match computeType.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending computeType: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    #[cfg(feature = "async_api")]
    {
        return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    }
    #[cfg(not(feature = "async_api"))]
    {
        match result.recv(channel_receiver) {
            Ok(_) => {}
            Err(e) => {
                error!("Error receiving result: {:?}", e);
            }
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        return result;
    }
}

#[no_mangle]
pub extern "C" fn cudnnGetConvolutionForwardAlgorithm_v7(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    requestAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t 
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnGetConvolutionForwardAlgorithm_v7",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1520;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let mut result: cudnnStatus_t = Default::default();
    match handle.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending handle: {:?}", e);
        }
    }
    match xDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending xDesc: {:?}", e);
        }
    }
    match wDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending wDesc: {:?}", e);
        }
    }
    match convDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending convDesc: {:?}", e);
        }
    }
    match yDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending yDesc: {:?}", e);
        }
    }
    match requestAlgoCount.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending requestAlgoCount: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    let mut returnedAlgoCount_: c_int = Default::default();
    let mut perfResults_: Vec<cudnnConvolutionFwdAlgoPerf_t> = Default::default();
    match returnedAlgoCount_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{*returnedAlgoCount = returnedAlgoCount_};
        }
        Err(e) => {
            error!("Error receiving returnedAlgoCount: {:?}", e);
        }
    }
    match perfResults_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{
                std::ptr::copy_nonoverlapping(
                    perfResults_.as_ptr(),
                    perfResults,
                    returnedAlgoCount_ as usize 
                );
            };
        }
        Err(e) => {
            error!("Error receiving perfResults: {:?}", e);
        }
    }
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}


#[no_mangle]
pub extern "C" fn cudnnConvolutionForward(
    handle: cudnnHandle_t,
    alpha: *const c_void,               // stored in host
    xDesc: cudnnTensorDescriptor_t,
    x: MemPtr,
    wDesc: cudnnFilterDescriptor_t,
    w: MemPtr,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workSpace: MemPtr,
    workSpaceSizeInBytes: size_t,
    beta: *const c_void,                // stored in host
    yDesc: cudnnTensorDescriptor_t,
    y: MemPtr,
) -> cudnnStatus_t {
    assert_eq!(true, *ENABLE_LOG);
    info!("[{}:{}] cudnnConvolutionForward", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    let proc_id = 1521;
    let mut result: cudnnStatus_t = Default::default();
    // process alpha and beta
    // assume that the datatype is double
    let alpha_ = unsafe {
        let slice = std::slice::from_raw_parts(alpha as *const f64, 1);
        slice[0]
    };
    let beta_ = unsafe {
        let slice = std::slice::from_raw_parts(beta as *const f64, 1);
        slice[0]
    };

    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match handle.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send handle: {:?}", e),
    }
    match alpha_.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send alpha: {:?}", e),
    }
    match xDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send xDesc: {:?}", e),
    }
    match x.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send x: {:?}", e),
    }
    match wDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send wDesc: {:?}", e),
    }
    match w.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send w: {:?}", e),
    }
    match convDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send convDesc: {:?}", e),
    }
    match algo.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send algo: {:?}", e),
    }
    match workSpace.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send workSpace: {:?}", e),
    }
    match workSpaceSizeInBytes.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send workSpaceSizeInBytes: {:?}", e),
    }
    match beta_.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send beta: {:?}", e),
    }
    match yDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send yDesc: {:?}", e),
    }
    match y.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send y: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    #[cfg(feature = "async_api")]
    {
        return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
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

#[no_mangle]
pub extern "C" fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut size_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1522;
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let mut result: cudnnStatus_t = Default::default();
    match handle.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending handle: {:?}", e)
    }
    match mode.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending mode: {:?}", e)
    }
    match bnOps.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending bnOps: {:?}", e)
    }
    match xDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending xDesc: {:?}", e)
    }
    match zDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending zDesc: {:?}", e)
    }
    match yDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending yDesc: {:?}", e)
    }
    match bnScaleBiasMeanVarDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending bnScaleBiasMeanVarDesc: {:?}", e)
    }
    match activationDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending activationDesc: {:?}", e)
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    // receive sizeInBytes from server
    let mut sizeInBytes_: size_t = Default::default();
    match sizeInBytes_.recv(channel_receiver){
        Ok(()) => {
            unsafe{*sizeInBytes = sizeInBytes_};
        }
        Err(e) => {
            error!("failed to receiving sizeInBytes: {:?}", e);
        }
    }
    match result.recv(channel_receiver){
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result 
}

#[no_mangle]
pub extern "C" fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    activationDesc: cudnnActivationDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut size_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnGetBatchNormalizationTrainingExReserveSpaceSize",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1523;
    let mut result: cudnnStatus_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    match handle.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending handle: {:?}", e)
    }
    match mode.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending mode: {:?}", e)
    }
    match bnOps.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending bnOps: {:?}", e)
    }
    match activationDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending activationDesc: {:?}", e)
    }
    match xDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => error!("Error sending xDesc: {:?}", e)
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    // receive sizeInBytes from server
    let mut sizeInBytes_: size_t = Default::default();
    match sizeInBytes_.recv(channel_receiver){
        Ok(()) => {
            unsafe{*sizeInBytes = sizeInBytes_};
        }
        Err(e) => {
            error!("failed to receiving sizeInBytes: {:?}", e);
        }
    }
    match result.recv(channel_receiver){
        Ok(()) => {}
        Err(e) => {
            error!("failed to receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnBatchNormalizationForwardTrainingEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    alpha: MemPtr,                          // assumed as double
    beta: MemPtr,                           // assumed as double
    xDesc: cudnnTensorDescriptor_t,
    xData: MemPtr,
    zDesc: cudnnTensorDescriptor_t,
    zData: MemPtr,
    yDesc: cudnnTensorDescriptor_t,
    yData: MemPtr,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScaleData: MemPtr,
    bnBiasData: MemPtr,
    exponentialAverageFactor: f64,
    resultRunningMeanData: MemPtr,
    resultRunningVarianceData: MemPtr,
    epsilon: f64,
    saveMean: MemPtr,
    saveInvVariance: MemPtr,
    activationDesc: cudnnActivationDescriptor_t,
    workSpace: MemPtr,
    workSpaceSizeInBytes: size_t,
    reserveSpace: MemPtr,
    reserveSpaceSizeInBytes: size_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnBatchNormalizationForwardTrainingEx",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1524;
    let mut result: cudnnStatus_t = Default::default();
    // process alpha and beta
    let alpha_ = unsafe {
        let slice = std::slice::from_raw_parts(alpha as *const f64, 1);
        slice[0]
    };
    let beta_ = unsafe {
        let slice = std::slice::from_raw_parts(beta as *const f64, 1);
        slice[0]
    };
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match handle.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send handle: {:?}", e),
    }
    match mode.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send mode: {:?}", e),
    }
    match bnOps.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send bnOps: {:?}", e),
    }
    match alpha_.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send alpha: {:?}", e),
    }
    match beta_.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send beta: {:?}", e),
    }
    match xDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send xDesc: {:?}", e),
    }
    match xData.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send xData: {:?}", e),
    }
    match zDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send zDesc: {:?}", e),
    }
    match zData.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send zData: {:?}", e),
    }
    match yDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send yDesc: {:?}", e),
    }
    match yData.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send yData: {:?}", e),
    }
    match bnScaleBiasMeanVarDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send bnScaleBiasMeanVarDesc: {:?}", e),
    }
    match bnScaleData.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send bnScaleData: {:?}", e),
    }
    match bnBiasData.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send bnBiasData: {:?}", e),
    }
    match exponentialAverageFactor.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send exponentialAverageFactor: {:?}", e),
    }
    match resultRunningMeanData.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send resultRunningMean: {:?}", e),
    }
    match resultRunningVarianceData.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send resultRunningVariance: {:?}", e),
    }
    match epsilon.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send epsilon: {:?}", e),
    }
    match saveMean.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send saveMean: {:?}", e),
    }
    match saveInvVariance.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send saveInvVariance: {:?}", e),
    }
    match activationDesc.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send activationDesc: {:?}", e),
    }
    match workSpace.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send workSpace: {:?}", e),
    }
    match workSpaceSizeInBytes.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send workSpaceSizeInBytes: {:?}", e),
    }
    match reserveSpace.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send reserveSpace: {:?}", e),
    }
    match reserveSpaceSizeInBytes.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send reserveSpaceSizeInBytes: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
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

#[no_mangle]
pub extern "C" fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    dzDesc: cudnnTensorDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut size_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnGetBatchNormalizationBackwardExWorkspaceSize",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1525;
    let mut result: cudnnStatus_t = Default::default();
    if let Err(e) = proc_id.send(channel_sender) {
        error!("Error sending proc_id: {:?}", e);
    }
    if let Err(e) = handle.send(channel_sender) {
        error!("Error sending handle: {:?}", e);
    }
    if let Err(e) = mode.send(channel_sender) {
        error!("Error sending mode: {:?}", e);
    }
    if let Err(e) = bnOps.send(channel_sender) {
        error!("Error sending bnOps: {:?}", e);
    }
    if let Err(e) = xDesc.send(channel_sender) {
        error!("Error sending xDesc: {:?}", e);
    }
    if let Err(e) = yDesc.send(channel_sender) {
        error!("Error sending yDesc: {:?}", e);
    }
    if let Err(e) = dyDesc.send(channel_sender) {
        error!("Error sending dyDesc: {:?}", e);
    }
    if let Err(e) = dzDesc.send(channel_sender) {
        error!("Error sending dzDesc: {:?}", e);
    }
    if let Err(e) = dxDesc.send(channel_sender) {
        error!("Error sending dxDesc: {:?}", e);
    }
    if let Err(e) = dBnScaleBiasDesc.send(channel_sender) {
        error!("Error sending dBnScaleBiasDesc: {:?}", e);
    }
    if let Err(e) = activationDesc.send(channel_sender) {
        error!("Error sending activationDesc: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    // receive sizeInBytes from server
    let mut sizeInBytes_: size_t = Default::default();
    if let Err(e) = sizeInBytes_.recv(channel_receiver) {
        error!("failed to receiving sizeInBytes: {:?}", e);
    } else {
        unsafe{*sizeInBytes = sizeInBytes_};
    }
    if let Err(e) = result.recv(channel_receiver) {
        error!("failed to receiving result: {:?}", e);
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnBatchNormalizationBackwardEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    alphaDataDiff: *const ::std::os::raw::c_void,       // assume is double
    betaDataDiff: *const ::std::os::raw::c_void,       // assume is double
    alphaParamDiff: *const ::std::os::raw::c_void,       // assume is double
    betaParamDiff: *const ::std::os::raw::c_void,       // assume is double
    xDesc: cudnnTensorDescriptor_t,
    xData: MemPtr,
    yDesc: cudnnTensorDescriptor_t,
    yData: MemPtr,
    dyDesc: cudnnTensorDescriptor_t,
    dyData: MemPtr,
    dzDesc: cudnnTensorDescriptor_t,
    dzData: MemPtr,
    dxDesc: cudnnTensorDescriptor_t,
    dxData: MemPtr,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    bnScaleData: MemPtr,
    bnBiasData: MemPtr,
    dBnScaleData: MemPtr,
    dBnBiasData: MemPtr,
    epsilon: f64,
    savedMean: MemPtr,
    savedInvVariance: MemPtr,
    activationDesc: cudnnActivationDescriptor_t,
    workSpace: MemPtr,
    workSpaceSizeInBytes: size_t,
    reserveSpace: MemPtr,
    reserveSpaceSizeInBytes: size_t,
) -> cudnnStatus_t {
    info!("[{}:{}] cudnnBatchNormalizationBackwardEx", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1526;
    let mut result: cudnnStatus_t = Default::default();
    // process alpha and beta
    let alphaDataDiff_ = unsafe{
        let slice = std::slice::from_raw_parts(alphaDataDiff as *const f64, 1);
        slice[0]
    };
    let betaDataDiff_ = unsafe {
        let slice = std::slice::from_raw_parts(betaDataDiff as *const f64, 1);
        slice[0]
    };
    let alphaParamDiff_ = unsafe {
        let slice = std::slice::from_raw_parts(alphaParamDiff as *const f64, 1);
        slice[0]
    };
    let betaParamDiff_ = unsafe {
        let slice = std::slice::from_raw_parts(betaParamDiff as *const f64, 1);
        slice[0]
    };
    // send data
    if let Err(e) = proc_id.send(channel_sender) {
        error!("Error sending proc_id: {:?}", e);  
    }
    if let Err(e) = handle.send(channel_sender) {
        error!("Error sending handle: {:?}", e);
    }
    if let Err(e) = mode.send(channel_sender) {
        error!("Error sending mode: {:?}", e);
    }
    if let Err(e) = bnOps.send(channel_sender) {
        error!("Error sending bnOps: {:?}", e);
    }
    if let Err(e) = alphaDataDiff_.send(channel_sender) {
        error!("Error sending alphaDataDiff: {:?}", e);
    }
    if let Err(e) = betaDataDiff_.send(channel_sender) {
        error!("Error sending betaDataDiff: {:?}", e);
    }
    if let Err(e) = alphaParamDiff_.send(channel_sender) {
        error!("Error sending alphaParamDiff: {:?}", e);
    }
    if let Err(e) = betaParamDiff_.send(channel_sender) {
        error!("Error sending betaParamDiff: {:?}", e);
    }
    if let Err(e) = xDesc.send(channel_sender) {
        error!("Error sending xDesc: {:?}", e);
    }
    if let Err(e) = xData.send(channel_sender) {
        error!("Error sending xData: {:?}", e);
    }
    if let Err(e) = yDesc.send(channel_sender) {
        error!("Error sending yDesc: {:?}", e);
    }
    if let Err(e) = yData.send(channel_sender) {
        error!("Error sending yData: {:?}", e);
    }
    if let Err(e) = dyDesc.send(channel_sender) {
        error!("Error sending dyDesc: {:?}", e);
    }
    if let Err(e) = dyData.send(channel_sender) {
        error!("Error sending dyData: {:?}", e);
    }
    if let Err(e) = dzDesc.send(channel_sender) {
        error!("Error sending dzDesc: {:?}", e);
    }
    if let Err(e) = dzData.send(channel_sender) {
        error!("Error sending dzData: {:?}", e);
    }
    if let Err(e) = dxDesc.send(channel_sender) {
        error!("Error sending dxDesc: {:?}", e);
    }
    if let Err(e) = dxData.send(channel_sender) {
        error!("Error sending dxData: {:?}", e);
    }
    if let Err(e) = dBnScaleBiasDesc.send(channel_sender) {
        error!("Error sending dBnScaleBiasDesc: {:?}", e);
    }
    if let Err(e) = bnScaleData.send(channel_sender) {
        error!("Error sending bnScaleData: {:?}", e);
    }
    if let Err(e) = bnBiasData.send(channel_sender) {
        error!("Error sending bnBiasData: {:?}", e);
    }
    if let Err(e) = dBnScaleData.send(channel_sender) {
        error!("Error sending dBnScaleData: {:?}", e);
    }
    if let Err(e) = dBnBiasData.send(channel_sender) {
        error!("Error sending dBnBiasData: {:?}", e);
    }
    if let Err(e) = epsilon.send(channel_sender) {
        error!("Error sending epsilon: {:?}", e);
    }
    if let Err(e) = savedMean.send(channel_sender) {
        error!("Error sending savedMean: {:?}", e);
    }
    if let Err(e) = savedInvVariance.send(channel_sender) {
        error!("Error sending savedInvVariance: {:?}", e);
    }
    if let Err(e) = activationDesc.send(channel_sender) {
        error!("Error sending activationDesc: {:?}", e);
    }
    if let Err(e) = workSpace.send(channel_sender) {
        error!("Error sending workSpace: {:?}", e);
    }
    if let Err(e) = workSpaceSizeInBytes.send(channel_sender) {
        error!("Error sending workSpaceSizeInBytes: {:?}", e);
    }
    if let Err(e) = reserveSpace.send(channel_sender) {
        error!("Error sending reserveSpace: {:?}", e);
    }
    if let Err(e) = reserveSpaceSizeInBytes.send(channel_sender) {
        error!("Error sending reserveSpaceSizeInBytes: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    if let Err(e) = result.recv(channel_receiver) {
        error!("failed to receive result: {:?}", e);
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    requestAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t 
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnGetConvolutionBackwardDataAlgorithm_v7",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1527;
    match proc_id.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending proc_id: {:?}", e);
        }
    }
    let mut result: cudnnStatus_t = Default::default();
    match handle.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending handle: {:?}", e);
        }
    }
    match wDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending wDesc: {:?}", e);
        }
    }
    match dyDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending xDesc: {:?}", e);
        }
    }
    match convDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending convDesc: {:?}", e);
        }
    }
    match dxDesc.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending yDesc: {:?}", e);
        }
    }
    match requestAlgoCount.send(channel_sender) {
        Ok(_) => {}
        Err(e) => {
            error!("Error sending requestAlgoCount: {:?}", e);
        }
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    let mut returnedAlgoCount_: c_int = Default::default();
    let mut perfResults_: Vec<cudnnConvolutionBwdDataAlgoPerf_t> = Default::default();
    match returnedAlgoCount_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{*returnedAlgoCount = returnedAlgoCount_};
        }
        Err(e) => {
            error!("Error receiving returnedAlgoCount: {:?}", e);
        }
    }
    match perfResults_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{
                std::ptr::copy_nonoverlapping(
                    perfResults_.as_ptr(),
                    perfResults,
                    returnedAlgoCount_ as usize 
                );
            };
        }
        Err(e) => {
            error!("Error receiving perfResults: {:?}", e);
        }
    }
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => {
            error!("Error receiving result: {:?}", e);
        }
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnConvolutionBackwardData(
    handle: cudnnHandle_t,
    alpha: *const c_void,               // stored in host
    wDesc: cudnnFilterDescriptor_t,
    w: MemPtr,
    dyDesc: cudnnTensorDescriptor_t,
    dy: MemPtr,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    workSpace: MemPtr,
    workSpaceSizeInBytes: size_t,
    beta: *const c_void,                // stored in host
    dxDesc: cudnnTensorDescriptor_t,
    dx: MemPtr,
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnConvolutionBackwardData",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1528;
    let mut result: cudnnStatus_t = Default::default();
    // process alpha and beta
    // assume that the datatype is double
    let alpha_ = unsafe {
        let slice = std::slice::from_raw_parts(alpha as *const f64, 1);
        slice[0]
    };
    let beta_ = unsafe {
        let slice = std::slice::from_raw_parts(beta as *const f64, 1);
        slice[0]
    };
    if let Err(e) = proc_id.send(channel_sender) {
        panic!("failed to send proc_id: {:?}", e);
    }
    if let Err(e) = handle.send(channel_sender) {
        panic!("failed to send handle: {:?}", e);
    }
    if let Err(e) = alpha_.send(channel_sender) {
        panic!("failed to send alpha: {:?}", e);
    }
    if let Err(e) = wDesc.send(channel_sender) {
        panic!("failed to send wDesc: {:?}", e);
    }
    if let Err(e) = w.send(channel_sender) {
        panic!("failed to send w: {:?}", e);
    }
    if let Err(e) = dyDesc.send(channel_sender) {
        panic!("failed to send dyDesc: {:?}", e);
    }
    if let Err(e) = dy.send(channel_sender) {
        panic!("failed to send dy: {:?}", e);
    }
    if let Err(e) = convDesc.send(channel_sender) {
        panic!("failed to send convDesc: {:?}", e);
    }
    if let Err(e) = algo.send(channel_sender) {
        panic!("failed to send algo: {:?}", e);
    }
    if let Err(e) = workSpace.send(channel_sender) {
        panic!("failed to send workSpace: {:?}", e);
    }
    if let Err(e) = workSpaceSizeInBytes.send(channel_sender) {
        panic!("failed to send workSpaceSizeInBytes: {:?}", e);
    }
    if let Err(e) = beta_.send(channel_sender) {
        panic!("failed to send beta: {:?}", e);
    }
    if let Err(e) = dxDesc.send(channel_sender) {
        panic!("failed to send dxDesc: {:?}", e);
    }
    if let Err(e) = dx.send(channel_sender) {
        panic!("failed to send dx: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    if let Err(e) = result.recv(channel_receiver) {
        panic!("failed to receive result: {:?}", e);
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dwDesc: cudnnFilterDescriptor_t,
    requestAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnGetConvolutionBackwardFilterAlgorithm_v7",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1529;
    if let Err(e) = proc_id.send(channel_sender) {
        error!("Error sending proc_id: {:?}", e);
    }
    let mut result: cudnnStatus_t = Default::default();
    if let Err(e) = handle.send(channel_sender) {
        error!("Error sending handle: {:?}", e);
    }
    if let Err(e) = xDesc.send(channel_sender) {
        error!("Error sending xDesc: {:?}", e);
    }
    if let Err(e) = dyDesc.send(channel_sender) {
        error!("Error sending dyDesc: {:?}", e);
    }
    if let Err(e) = convDesc.send(channel_sender) {
        error!("Error sending convDesc: {:?}", e);
    }
    if let Err(e) = dwDesc.send(channel_sender) {
        error!("Error sending dwDesc: {:?}", e);
    }
    if let Err(e) = requestAlgoCount.send(channel_sender) {
        error!("Error sending requestAlgoCount: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    let mut returnedAlgoCount_: c_int = Default::default();
    let mut perfResults_: Vec<cudnnConvolutionBwdFilterAlgoPerf_t> = Default::default();
    if let Err(e) = returnedAlgoCount_.recv(channel_receiver) {
        error!("Error receiving returnedAlgoCount: {:?}", e);
    } else {
        unsafe{*returnedAlgoCount = returnedAlgoCount_};
    }
    if let Err(e) = perfResults_.recv(channel_receiver) {
        error!("Error receiving perfResults: {:?}", e);
    } else {
        unsafe{
            std::ptr::copy_nonoverlapping(
                perfResults_.as_ptr(),
                perfResults,
                returnedAlgoCount_ as usize 
            );
        }
    }
    if let Err(e) = result.recv(channel_receiver) {
        error!("Error receiving result: {:?}", e);
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnConvolutionBackwardFilter(
    handle: cudnnHandle_t,
    alpha: MemPtr,
    xDesc: cudnnTensorDescriptor_t,
    x: MemPtr,
    dyDesc: cudnnTensorDescriptor_t,
    dy: MemPtr,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    workSpace: MemPtr,
    workSpaceSizeInBytes: size_t,
    beta: MemPtr,
    dwDesc: cudnnFilterDescriptor_t,
    dw: MemPtr,
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnConvolutionBackwardFilter",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1530;
    let mut result: cudnnStatus_t = Default::default();
    // process alpha and beta
    // assume that the datatype is double
    let alpha_ = unsafe {
        let slice = std::slice::from_raw_parts(alpha as *const f64, 1);
        slice[0]
    };
    let beta_ = unsafe {
        let slice = std::slice::from_raw_parts(beta as *const f64, 1);
        slice[0]
    };
    if let Err(e) = proc_id.send(channel_sender) {
        panic!("failed to send proc_id: {:?}", e);
    }
    if let Err(e) = handle.send(channel_sender) {
        panic!("failed to send handle: {:?}", e);
    }
    if let Err(e) = alpha_.send(channel_sender) {
        panic!("failed to send alpha: {:?}", e);
    }
    if let Err(e) = xDesc.send(channel_sender) {
        panic!("failed to send xDesc: {:?}", e);
    }
    if let Err(e) = x.send(channel_sender) {
        panic!("failed to send x: {:?}", e);
    }
    if let Err(e) = dyDesc.send(channel_sender) {
        panic!("failed to send dyDesc: {:?}", e);
    }
    if let Err(e) = dy.send(channel_sender) {
        panic!("failed to send dy: {:?}", e);
    }
    if let Err(e) = convDesc.send(channel_sender) {
        panic!("failed to send convDesc: {:?}", e);
    }
    if let Err(e) = algo.send(channel_sender) {
        panic!("failed to send algo: {:?}", e);
    }
    if let Err(e) = workSpace.send(channel_sender) {
        panic!("failed to send workSpace: {:?}", e);
    }
    if let Err(e) = workSpaceSizeInBytes.send(channel_sender) {
        panic!("failed to send workSpaceSizeInBytes: {:?}", e);
    }
    if let Err(e) = beta_.send(channel_sender) {
        panic!("failed to send beta: {:?}", e);
    }
    if let Err(e) = dwDesc.send(channel_sender) {
        panic!("failed to send dwDesc: {:?}", e);
    }
    if let Err(e) = dw.send(channel_sender) {
        panic!("failed to send dw: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    if let Err(e) = result.recv(channel_receiver) {
        panic!("failed to receive result: {:?}", e);
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    result
}

#[no_mangle]
pub extern "C" fn cudnnBatchNormalizationForwardInference(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: MemPtr,                  // on host memory, assumed as double
    beta: MemPtr,                   // on host memory, assumed as double
    xDesc: cudnnTensorDescriptor_t,
    x: MemPtr,
    yDesc: cudnnTensorDescriptor_t,
    y: MemPtr,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    bnScale: MemPtr,
    bnBias: MemPtr,
    estimatedMean: MemPtr,
    estimatedVariance: MemPtr,
    epsilon: f64,
) -> cudnnStatus_t {
    info!(
        "[{}:{}] cudnnBatchNormalizationForwardInference",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1531;
    let mut result: cudnnStatus_t = Default::default();
    // process alpha and beta
    let alpha_ = unsafe {
        let slice = std::slice::from_raw_parts(alpha as *const f64, 1);
        slice[0]
    };
    let beta_ = unsafe {
        let slice = std::slice::from_raw_parts(beta as *const f64, 1);
        slice[0]
    };
    if let Err(e) = proc_id.send(channel_sender) {
        panic!("failed to send proc_id: {:?}", e);
    }
    if let Err(e) = handle.send(channel_sender) {
        panic!("failed to send handle: {:?}", e);
    }
    if let Err(e) = mode.send(channel_sender) {
        panic!("failed to send mode: {:?}", e);
    }
    if let Err(e) = alpha_.send(channel_sender) {
        panic!("failed to send alpha: {:?}", e);
    }
    if let Err(e) = beta_.send(channel_sender) {
        panic!("failed to send beta: {:?}", e);
    }
    if let Err(e) = xDesc.send(channel_sender) {
        panic!("failed to send xDesc: {:?}", e);
    }
    if let Err(e) = x.send(channel_sender) {
        panic!("failed to send x: {:?}", e);
    }
    if let Err(e) = yDesc.send(channel_sender) {
        panic!("failed to send yDesc: {:?}", e);
    }
    if let Err(e) = y.send(channel_sender) {
        panic!("failed to send y: {:?}", e);
    }
    if let Err(e) = bnScaleBiasMeanVarDesc.send(channel_sender) {
        panic!("failed to send bnScaleBiasMeanVarDesc: {:?}", e);
    }
    if let Err(e) = bnScale.send(channel_sender) {
        panic!("failed to send bnScale: {:?}", e);
    }
    if let Err(e) = bnBias.send(channel_sender) {
        panic!("failed to send bnBias: {:?}", e);
    }
    if let Err(e) = estimatedMean.send(channel_sender) {
        panic!("failed to send estimatedMean: {:?}", e);
    }
    if let Err(e) = estimatedVariance.send(channel_sender) {
        panic!("failed to send estimatedVariance: {:?}", e);
    }
    if let Err(e) = epsilon.send(channel_sender) {
        panic!("failed to send epsilon: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    #[cfg(feature = "async_api")]
    {
        return cudnnStatus_t::CUDNN_STATUS_SUCCESS;
    }
    #[cfg(not(feature = "async_api"))]
    {
        // receive result from server
        if let Err(e) = result.recv(channel_receiver) {
            panic!("failed to receive result: {:?}", e);
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        return result;
    }
}