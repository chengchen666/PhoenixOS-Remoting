#![allow(non_snake_case)]
use super::*;
use cudasys::types::cuda::*;

gen_hijack!(
    300,
    "cuDevicePrimaryCtxGetState",
    "CUresult",
    "CUdevice",
    "*mut ::std::os::raw::c_uint",
    "*mut ::std::os::raw::c_int"
);
