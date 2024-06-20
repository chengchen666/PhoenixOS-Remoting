#![allow(non_snake_case)]

use super::*;
use cudasys::cublas::*;

gen_exe!(
    "cublasDestroy_v2", 
    "cublasStatus_t", 
    "cublasHandle_t"
);

#[cfg(feature = "async_api")]
gen_exe_async!(
    "cublasSetStream_v2", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cudaStream_t"
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
    "cublasSetStream_v2", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cudaStream_t"
);

#[cfg(feature = "async_api")]
gen_exe_async!(
    "cublasSetMathMode", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cublasMath_t"
);
#[cfg(not(feature = "async_api"))]
gen_exe!(
    "cublasSetMathMode", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cublasMath_t"
);

gen_exe!(
    "cublasGetMathMode",
    "cublasStatus_t",
    "cublasHandle_t",
    "*mut cublasMath_t"
);

gen_exe!(
    "cublasGemmEx",
    "cublasStatus_t",
    "cublasHandle_t",
    "cublasOperation_t",
    "cublasOperation_t",
    "::std::os::raw::c_int",
    "::std::os::raw::c_int",
    "::std::os::raw::c_int",
    "MemPtr",
    "MemPtr",
    "cudaDataType",
    "::std::os::raw::c_int",
    "MemPtr",
    "cudaDataType",
    "::std::os::raw::c_int",
    "MemPtr",
    "MemPtr",
    "cudaDataType",
    "::std::os::raw::c_int",
    "cublasComputeType_t",
    "cublasGemmAlgo_t"
);