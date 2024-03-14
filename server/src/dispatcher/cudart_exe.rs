#![allow(non_snake_case)]
use super::*;

gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_exe!("cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_exe!(
    "cudaGetDeviceCount",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
gen_exe!("cudaGetLastError", "cudaError_t");
gen_exe!("cudaPeekAtLastError", "cudaError_t");
gen_exe!("cudaStreamSynchronize", "cudaError_t", "cudaStream_t");
gen_exe!("cudaMalloc", "cudaError_t", "*mut MemPtr", "usize");
// gen_exe!(
//     "cudaMemcpy",
//     "cudaError_t",
//     "MemPtr",
//     "MemPtr",
//     "usize",
//     "cudaMemcpyKind"
// );
gen_exe!("cudaFree", "cudaError_t", "MemPtr");
gen_exe!(
    "cudaStreamIsCapturing",
    "cudaError_t",
    "cudaStream_t",
    "*mut cudaStreamCaptureStatus"
);
gen_exe!(
    "cudaGetDeviceProperties",
    "cudaError_t",
    "*mut cudaDeviceProp",
    "::std::os::raw::c_int"
);
// gen_exe!(
//     "__cudaRegisterFatBinary",
//     "MemPtr",
//     "*const ::std::os::raw::c_void"
// );
// gen_exe!("__cudaUnregisterFatBinary", "null", "MemPtr");
// gen_exe!(
//     "__cudaRegisterFunction",
//     "MemPtr",
//     "MemPtr",
//     "*const ::std::os::raw::c_void",
//     "*const ::std::os::raw::c_char",
//     "::std::os::raw::c_int",
//     "::std::os::raw::c_int",
//     "MemPtr",
//     "MemPtr"
// );
