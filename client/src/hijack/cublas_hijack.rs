use super::*;
use cudasys::types::cublas::*;
use std::os::raw::*;

#[cuda_hook_hijack(proc_id = 2001)]
fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;

#[cuda_hook_hijack(proc_id = 2002, async_api)]
fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;

#[cuda_hook_hijack(proc_id = 2003, async_api)]
fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;

#[cuda_hook_hijack(proc_id = 2006)]
fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> cublasStatus_t;

// gen_hijack!(
//     2007,
//     "cublasGemmEx",
//     "cublasStatus_t",
//     "cublasHandle_t",
//     "cublasOperation_t",
//     "cublasOperation_t",
//     "::std::os::raw::c_int",
//     "::std::os::raw::c_int",
//     "::std::os::raw::c_int",
//     "MemPtr",
//     "MemPtr",
//     "cudaDataType",
//     "::std::os::raw::c_int",
//     "MemPtr",
//     "cudaDataType",
//     "::std::os::raw::c_int",
//     "MemPtr",
//     "MemPtr",
//     "cudaDataType",
//     "::std::os::raw::c_int",
//     "cublasComputeType_t",
//     "cublasGemmAlgo_t"
// );

#[cuda_hook_hijack(proc_id = 2009)]
fn cublasSetWorkspace_v2(
    handle: cublasHandle_t,
    #[device] workspace: *mut c_void,
    workspaceSizeInBytes: usize,
) -> cublasStatus_t;
