#![allow(non_snake_case)]
use super::*;
use cudasys::cudart::*;

gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_exe!("cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_exe!(
    "cudaGetDeviceCount",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
gen_exe!("cudaGetLastError", "cudaError_t");
gen_exe!("cudaPeekAtLastError", "cudaError_t");
#[cfg(feature = "async_api")]
gen_exe_async!("cudaStreamSynchronize", "cudaError_t", "cudaStream_t");
#[cfg(not(feature = "async_api"))]
gen_exe!("cudaStreamSynchronize", "cudaError_t", "cudaStream_t");
// gen_exe!("cudaMalloc", "cudaError_t", "*mut MemPtr", "size_t");
// gen_exe!(
//     "cudaMemcpy",
//     "cudaError_t",
//     "MemPtr",
//     "MemPtr",
//     "size_t",
//     "cudaMemcpyKind"
// );
// gen_exe!("cudaFree", "cudaError_t", "MemPtr");
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
gen_exe!(
    "cudaMemGetInfo", 
    "cudaError_t", 
    "*mut size_t", 
    "*mut size_t"
);
// gen_exe!(
//     "__cudaRegisterFatBinary",
//     "MemPtr",
//     "*const ::std::os::raw::c_void"
// );
// gen_exe!("__cudaUnregisterFatBinary", "null", "MemPtr");
// gen_exe!(
//     "__cudaRegisterFunction",
//     "null",
//     "MemPtr",
//     "MemPtr",
//     "*mut ::std::os::raw::c_char",
//     "*const ::std::os::raw::c_char",
//     "::std::os::raw::c_int",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr",
//     "MemPtr"
// );
// gen_exe!(
//     "__cudaRegisterVar",
//     "null",
//     "MemPtr",
//     "MemPtr",
//     "*const ::std::os::raw::c_char",
//     "*const ::std::os::raw::c_char",
//     "::std::os::raw::c_int",
//     "usize",
//     "::std::os::raw::c_int",
//     "::std::os::raw::c_int"
// );
// gen_exe!(
//     "cudaLaunchKernel",
//     "cudaError_t",
//     "*const ::std::os::raw::c_void",
//     "dim3",
//     "dim3",
//     "*mut *mut ::std::os::raw::c_void",
//     "usize",
//     "cudaStream_t"
// );

gen_exe!(
    "cudaDeviceGetStreamPriorityRange",
    "cudaError_t",
    "*mut ::std::os::raw::c_int",
    "*mut ::std::os::raw::c_int"
);