use crate::types::cudart::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_hook(proc_id = 0)]
fn cudaGetDevice(device: *mut c_int) -> cudaError_t {
    'client_before_send: {
        #[cfg(feature = "local")]
        if let Some(val) = client.cuda_device {
            unsafe {
                *device = val;
            }
            return cudaError_t::cudaSuccess;
        }
    }
    'client_after_recv: {
        #[cfg(feature = "local")]
        {
            client.cuda_device = Some(*device);
        }
    }
}

#[cuda_hook(proc_id = 1)]
fn cudaSetDevice(device: c_int) -> cudaError_t {
    'client_before_send: {
        #[cfg(feature = "local")]
        {
            client.cuda_device = None;
        }
    }
}

#[cuda_hook(proc_id = 2)]
fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;

#[cuda_hook(proc_id = 3)]
fn cudaGetLastError() -> cudaError_t;

#[cuda_hook(proc_id = 4)]
fn cudaPeekAtLastError() -> cudaError_t;

#[cuda_hook(proc_id = 5)]
fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 6)]
fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;

#[cuda_custom_hook(proc_id = 7)]
fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t;

#[cuda_custom_hook] // calls cudaMemcpy (wrong implementation)
fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: cudaMemcpyKind,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 8, async_api)]
fn cudaFree(#[device] devPtr: *mut c_void) -> cudaError_t;

#[cuda_hook(proc_id = 9)]
fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> cudaError_t;

// This function is hidden and superseded by `cudaGetDeviceProperties_v2` in CUDA 12.
// The change is that `cudaDeviceProp` grew bigger. We don't hook it in CUDA 12
// to prevent reading or writing past the end of allocated memory when sending or receiving data.
#[cuda_hook(proc_id = 10, max_cuda_version = 11)]
fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

// 11 => unimplemented!("cudaMallocManaged")

#[cuda_hook(proc_id = 12)]
fn cudaPointerGetAttributes(
    attributes: *mut cudaPointerAttributes,
    #[device] ptr: *const c_void,
) -> cudaError_t;

#[cuda_custom_hook] // local, proc_id was 13
fn cudaHostAlloc(pHost: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 14)]
fn cudaFuncGetAttributes(
    attr: *mut cudaFuncAttributes,
    #[device] func: *const c_void,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 100)]
fn __cudaRegisterFatBinary(fatCubin: *mut c_void) -> *mut *mut c_void;

#[cuda_custom_hook] // local
fn __cudaRegisterFatBinaryEnd(fatCubinHandle: *mut *mut c_void);

#[cuda_custom_hook] // local, proc_id was 101
fn __cudaUnregisterFatBinary(fatCubinHandle: *mut *mut c_void);

#[cuda_custom_hook(proc_id = 102)]
fn __cudaRegisterFunction(
    fatCubinHandle: *mut *mut c_void,
    hostFun: *const c_char,
    deviceFun: *mut c_char,
    deviceName: *const c_char,
    thread_limit: c_int,
    tid: *mut uint3,
    bid: *mut uint3,
    bDim: *mut dim3,
    gDim: *mut dim3,
    wSize: *mut c_int,
);

#[cuda_custom_hook(proc_id = 103)]
fn __cudaRegisterVar(
    fatCubinHandle: *mut *mut c_void,
    hostVar: *mut c_char,
    deviceAddress: *mut c_char,
    deviceName: *const c_char,
    ext: c_int,
    size: usize,
    constant: c_int,
    global: c_int,
);

#[cuda_custom_hook(proc_id = 200)]
fn cudaLaunchKernel(
    func: *const c_void,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook(proc_id = 15)]
fn cudaDeviceGetStreamPriorityRange(
    leastPriority: *mut c_int,
    greatestPriority: *mut c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 16, async_api)]
fn cudaMemsetAsync(
    #[device] devPtr: *mut c_void,
    value: c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_custom_hook(proc_id = 17)]
fn cudaGetErrorString(error: cudaError_t) -> *const c_char;

#[cuda_hook(proc_id = 18)]
fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

#[cuda_custom_hook] // local
fn __cudaPushCallConfiguration(
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: *mut CUstream_st,
) -> c_uint;

#[cuda_custom_hook] // local
fn __cudaPopCallConfiguration(
    gridDim: *mut dim3,
    blockDim: *mut dim3,
    sharedMem: *mut usize,
    stream: *mut c_void,
) -> cudaError_t;

#[cuda_hook(proc_id = 19, min_cuda_version = 12)]
fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook(proc_id = 20)]
fn cudaStreamCreateWithPriority(
    pStream: *mut cudaStream_t,
    flags: c_uint,
    priority: c_int,
) -> cudaError_t;

#[cuda_hook(proc_id = 21)]
fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 22)]
fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;

#[cuda_hook(proc_id = 23)]
fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) -> cudaError_t;

#[cuda_hook(proc_id = 24)]
fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
