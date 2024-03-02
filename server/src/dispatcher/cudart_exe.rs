#![allow(non_snake_case)]
use super::*;

gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_exe!("cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_exe!("cudaGetDeviceCount", "cudaError_t", "*mut ::std::os::raw::c_int");
