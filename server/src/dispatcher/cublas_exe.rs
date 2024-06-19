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