#![allow(non_snake_case)]
use super::*;
use cudasys::cuda::*;

gen_exe!(
    "cuDevicePrimaryCtxGetState",
    "CUresult",
    "CUdevice",
    "*mut ::std::os::raw::c_uint",
    "*mut ::std::os::raw::c_int"
);

gen_exe!(
    "cuDriverGetVersion",
    "CUresult",
    "*mut ::std::os::raw::c_int"
);

gen_exe!(
    "cuInit",
    "CUresult",
    "::std::os::raw::c_uint"
);

gen_exe!("cuCtxGetCurrent", "CUresult", "*mut CUcontext");
