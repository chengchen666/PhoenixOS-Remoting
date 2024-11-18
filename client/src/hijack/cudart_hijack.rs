use super::*;
use cudasys::types::cudart::*;
use std::os::raw::*;

#[cuda_hook_hijack(proc_id = 0)]
fn cudaGetDevice(device: *mut c_int) -> cudaError_t {
    'client_before_send: {
        #[cfg(feature = "local")]
        if let Some(val) = get_local_info(proc_id as usize) {
            unsafe {
                *device = val as i32;
            }
            return cudaError_t::cudaSuccess;
        }
    }
    'client_after_recv: {
        #[cfg(feature = "local")]
        add_local_info(proc_id as usize, *device as usize);
    }
}

#[cuda_hook_hijack(proc_id = 1)]
fn cudaSetDevice(device: c_int) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 2)]
fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 3)]
fn cudaGetLastError() -> cudaError_t;

#[cuda_hook_hijack(proc_id = 4)]
fn cudaPeekAtLastError() -> cudaError_t;

#[cuda_hook_hijack(proc_id = 5)]
fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 6)]
fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;

// gen_hijack!(
//     7,
//     "cudaMemcpy",
//     "cudaError_t",
//     "MemPtr",
//     "MemPtr",
//     "size_t",
//     "cudaMemcpyKind"
// );

#[cuda_hook_hijack(proc_id = 8, async_api)]
fn cudaFree(#[device] devPtr: *mut c_void) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 9)]
fn cudaStreamIsCapturing(
    stream: cudaStream_t,
    pCaptureStatus: *mut cudaStreamCaptureStatus,
) -> cudaError_t;

// This function is hidden and superseded by `cudaGetDeviceProperties_v2` in CUDA 12.
// The change is that `cudaDeviceProp` grew bigger. We don't hook it in CUDA 12
// to prevent reading or writing past the end of allocated memory when sending or receiving data.
#[cuda_hook_hijack(proc_id = 10, max_cuda_version = 11)]
fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 12)]
fn cudaPointerGetAttributes(
    attributes: *mut cudaPointerAttributes,
    #[device] ptr: *const c_void,
) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 14)]
fn cudaFuncGetAttributes(
    attr: *mut cudaFuncAttributes,
    #[device] func: *const c_void,
) -> cudaError_t;

// gen_hijack!(
//     100,
//     "__cudaRegisterFatBinary",
//     "MemPtr",
//     "*const ::std::os::raw::c_void"
// );
// gen_hijack!(
//     101,
//     "__cudaUnregisterFatBinary",
//     "null",
//     "MemPtr"
// );
// gen_hijack!(
//     102,
//     "__cudaRegisterFunction",
//     "null",
//     "MemPtr",
//     "MemPtr",
//     "*mut ::std::os::raw::c_char",
//     "*const ::std::os::raw::c_char",
//     "::std::os::raw::c_int",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr"
// );
// gen_hijack!(
//     103,
//     "__cudaRegisterVar",
//     "null",
//     "MemPtr",
//     "MemPtr",
//     "*const ::std::os::raw::c_char",
//     "*const ::std::os::raw::c_char",
//     "::std::os::raw::c_int",
//     "usize",
//     "::std::os::raw::c_int",
//     "::std::os::raw::c_int"
// );
// gen_hijack!(
//     200,
//     "cudaLaunchKernel",
//     "cudaError_t",
//     "*const ::std::os::raw::c_void",
//     "dim3",
//     "dim3",
//     "*mut *mut ::std::os::raw::c_void",
//     "usize",
//     "cudaStream_t"
// );

#[cuda_hook_hijack(proc_id = 15)]
fn cudaDeviceGetStreamPriorityRange(
    leastPriority: *mut c_int,
    greatestPriority: *mut c_int,
) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 16, async_api)]
fn cudaMemsetAsync(
    #[device] devPtr: *mut c_void,
    value: c_int,
    count: usize,
    stream: cudaStream_t,
) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 17)]
fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;

#[cuda_hook_hijack(proc_id = 18, min_cuda_version = 12)]
fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
