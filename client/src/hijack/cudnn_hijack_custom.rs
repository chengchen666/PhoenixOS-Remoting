#![expect(non_snake_case)]

use super::*;
use cudasys::types::cudnn::*;
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeDouble = f64;

#[cuda_hook_hijack(proc_id = 1500)]
fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1501)]
fn cudnnCreateTensorDescriptor(
    tensorDesc: *mut cudnnTensorDescriptor_t,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1503)]
fn cudnnCreateActivationDescriptor(
    activationDesc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t {
}

// FIXME: this function used to send `alpha` and `beta` as `MemPtr`s but they are actually host
// pointers that need to be dereferenced. `handle` was also mistakenly typed `cudnnStatus_t`.
// Also, this function should be `async_api`.
/*
#[cuda_hook_hijack(proc_id = 1505)]
fn cudnnActivationForward(
    handle: cudnnHandle_t,
    activationDesc: cudnnActivationDescriptor_t,
    alpha: *const c_void,
    xDesc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t {
}
*/

#[cuda_hook_hijack(proc_id = 1509, async_api)]
fn cudnnSetTensorNdDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: c_int,
    #[host(len = nbDims)] dimA: *const c_int,
    #[host(len = nbDims)] strideA: *const c_int,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1511)]
fn cudnnCreateFilterDescriptor(
    filterDesc: *mut cudnnFilterDescriptor_t,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1513, async_api)]
fn cudnnSetFilterNdDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nbDims: c_int,
    #[host(len = nbDims)]
    filterDimA: *const c_int,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1514)]
fn cudnnCreateConvolutionDescriptor(
    convDesc: *mut cudnnConvolutionDescriptor_t,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1516, async_api)]
fn cudnnSetConvolutionNdDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLength: c_int,
    #[host(len = arrayLength)]
    padA: *const c_int,
    #[host(len = arrayLength)]
    filterStrideA: *const c_int,
    #[host(len = arrayLength)]
    dilationA: *const c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1520)]
fn cudnnGetConvolutionForwardAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1521, async_api)]
fn cudnnConvolutionForward(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1522)]
fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1523)]
fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    activationDesc: cudnnActivationDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1524, async_api)]
fn cudnnBatchNormalizationForwardTrainingEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    zDesc: cudnnTensorDescriptor_t,
    #[device] zData: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *mut c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    exponentialAverageFactor: f64,
    #[device] resultRunningMean: *mut c_void,
    #[device] resultRunningVariance: *mut c_void,
    epsilon: f64,
    #[device] resultSaveMean: *mut c_void,
    #[device] resultSaveInvVariance: *mut c_void,
    activationDesc: cudnnActivationDescriptor_t,
    #[device] workspace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1525)]
fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
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
    sizeInBytes: *mut usize,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1526, async_api)]
fn cudnnBatchNormalizationBackwardEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alphaDataDiff: *const HackedAssumeDouble,
    #[host] betaDataDiff: *const HackedAssumeDouble,
    #[host] alphaParamDiff: *const HackedAssumeDouble,
    #[host] betaParamDiff: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dyData: *const c_void,
    dzDesc: cudnnTensorDescriptor_t,
    #[device] dzData: *mut c_void,
    dxDesc: cudnnTensorDescriptor_t,
    #[device] dxData: *mut c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    #[device] bnScaleData: *const c_void,
    #[device] bnBiasData: *const c_void,
    #[device] dBnScaleData: *mut c_void,
    #[device] dBnBiasData: *mut c_void,
    epsilon: f64,
    #[device] savedMean: *const c_void,
    #[device] savedInvVariance: *const c_void,
    activationDesc: cudnnActivationDescriptor_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1527)]
fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1528, async_api)]
fn cudnnConvolutionBackwardData(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    dxDesc: cudnnTensorDescriptor_t,
    #[device] dx: *mut c_void,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1529)]
fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1530, async_api)]
fn cudnnConvolutionBackwardFilter(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    dwDesc: cudnnFilterDescriptor_t,
    #[device] dw: *mut c_void,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1531, async_api)]
fn cudnnBatchNormalizationForwardInference(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    #[device] estimatedMean: *const c_void,
    #[device] estimatedVariance: *const c_void,
    epsilon: f64,
) -> cudnnStatus_t {
}

#[cuda_hook_hijack(proc_id = 1533)]
fn cudnnGetConvolutionNdForwardOutputDim(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    nbDims: c_int,
    #[host(output, len = nbDims)]
    tensorOuputDimA: *mut c_int,
) -> cudnnStatus_t {
}

#[no_mangle]
pub extern "C" fn cudnnGetErrorString(
    error_status: cudnnStatus_t,
) -> *const c_char {
    info!(
        "[{}:{}] cudnnGetErrorString",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let proc_id = 1535;
    let mut result: Vec<u8> = Default::default();
    if let Err(e) = proc_id.send(channel_sender) {
        error!("Error sending proc_id: {:?}", e);
    }
    if let Err(e) = error_status.send(channel_sender) {
        error!("Error sending error_string: {:?}", e);
    }
    if let Err(e) = channel_sender.flush_out() {
        panic!("failed to send: {:?}", e);
    }
    if let Err(e) = result.recv(channel_receiver) {
        error!("Error receiving result: {:?}", e);
    }
    if let Err(e) = channel_receiver.recv_ts() {
        error!("Error receiving timestamp: {:?}", e);
    }
    let c_str = std::ffi::CString::new(result).unwrap();
    c_str.into_raw()
}