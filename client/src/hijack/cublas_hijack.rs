#![allow(non_snake_case)]
use super::*;
use cudasys::types::cublas::*;

gen_hijack!(
    2001,
    "cublasDestroy_v2", 
    "cublasStatus_t", 
    "cublasHandle_t"
);

gen_hijack!(
    2002,
    "cublasSetStream_v2", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cudaStream_t"
);

gen_hijack!(
    2003,
    "cublasSetMathMode", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cublasMath_t"
);