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

gen_hijack!(
    501,
    "cuDriverGetVersion",
    "CUresult",
    "*mut ::std::os::raw::c_int"
);

gen_hijack!(
    502,
    "cuInit",
    "CUresult",
    "::std::os::raw::c_uint"
);