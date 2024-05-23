#![allow(non_snake_case)]

use super::*;
use cudasys::cublas::*;

gen_exe!(
    "cublasDestroy_v2", 
    "cublasStatus_t", 
    "cublasHandle_t"
);

gen_exe!(
    "cublasSetStream_v2", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cudaStream_t"
);

gen_exe!(
    "cublasSetMathMode", 
    "cublasStatus_t", 
    "cublasHandle_t", 
    "cublasMath_t"
);