use super::*;
use cudasys::types::cudnn::*;
use std::os::raw::*;

#[cuda_hook_hijack(proc_id = 1502)]
fn cudnnSetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1504)]
fn cudnnSetActivationDescriptor(
    activationDesc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    reluNanOpt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1506)]
fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1507)]
fn cudnnSetConvolution2dDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: c_int,
    pad_w: c_int,
    u: c_int,
    v: c_int,
    dilation_h: c_int,
    dilation_w: c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1508, async_api)]
fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1510, async_api)]
fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1512, async_api)]
fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1515, async_api)]
fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1517, async_api)]
fn cudnnSetConvolutionGroupCount(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: c_int,
) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1518, async_api)]
fn cudnnSetConvolutionMathType(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: cudnnMathType_t,
) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1519)]
fn cudnnSetConvolutionReorderType(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: cudnnReorderType_t,
) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1532)]
fn cudnnSetFilter4dDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook_hijack(proc_id = 1534)]
fn cudnnGetConvolutionForwardWorkspaceSize(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;
