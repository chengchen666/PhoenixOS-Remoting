use super::*;
use cudasys::types::cublas::*;
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeFloat = f32;

#[cuda_hook_hijack(proc_id = 2000)]
fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;

#[cuda_hook_hijack(proc_id = 2004, async_api)]
fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const f32, // FIXME: safe until we support cublasSetPointerMode()
    #[device] A: *const f32,
    lda: c_int,
    #[device] B: *const f32,
    ldb: c_int,
    #[host] beta: *const f32,
    #[device] C: *mut f32,
    ldc: c_int,
) -> cublasStatus_t {
}

#[cuda_hook_hijack(proc_id = 2005, async_api)]
fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const f32,
    ldb: c_int,
    strideB: c_longlong,
    #[host] beta: *const f32,
    #[device] C: *mut f32,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> cublasStatus_t {
}

#[cuda_hook_hijack(proc_id = 2007, async_api)]
fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const HackedAssumeFloat,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    #[host] beta: *const HackedAssumeFloat,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
}

#[cuda_hook_hijack(proc_id = 2008, async_api)]
fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[host] alpha: *const HackedAssumeFloat,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    strideB: c_longlong,
    #[host] beta: *const HackedAssumeFloat,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
}
