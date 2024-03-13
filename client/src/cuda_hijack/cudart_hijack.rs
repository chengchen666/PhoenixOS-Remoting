#![allow(non_snake_case)]
use super::*;

gen_hijack!(
    0,
    "cudaGetDevice",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
gen_hijack!(1, "cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_hijack!(
    2,
    "cudaGetDeviceCount",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
gen_hijack!(3, "cudaGetLastError", "cudaError_t");
gen_hijack!(4, "cudaPeekAtLastError", "cudaError_t");
gen_hijack!(5, "cudaStreamSynchronize", "cudaError_t", "cudaStream_t");
gen_hijack!(6, "cudaMalloc", "cudaError_t", "*mut MemPtr", "usize");
// gen_hijack!(
//     7,
//     "cudaMemcpy",
//     "cudaError_t",
//     "MemPtr",
//     "MemPtr",
//     "usize",
//     "cudaMemcpyKind"
// );
gen_hijack!(8, "cudaFree", "cudaError_t", "*mut MemPtr");
gen_hijack!(
    9,
    "cudaStreamIsCapturing",
    "cudaError_t",
    "cudaStream_t",
    "*mut cudaStreamCaptureStatus"
);
gen_hijack!(
    10,
    "cudaGetDeviceProperties",
    "cudaError_t",
    "*mut cudaDeviceProp",
    "::std::os::raw::c_int"
);
