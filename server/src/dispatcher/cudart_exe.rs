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
// pub fn cudaGetLastErrorExe<T: CommChannel>(
//     channel_sender: &mut T,
//     channel_receiver: &mut T,
// ) {
//     info!("[{}:{}] {}", std::file!(), std::line!(), "cudaGetLastError");
//     let start = network::NsTimestamp::now();
//     match channel_receiver.recv_ts() {
//         Ok(()) => {}
//         Err(e) => {
//             panic!("failed to receive timestamp: {:?}", e)
//         }
//     }
//     let end = network::NsTimestamp::now();
//     let elapsed = (end.sec_timestamp - start.sec_timestamp) * 1000000000
//                 + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as i64;
//     info!("{}", elapsed);
    
//     let start = network::NsTimestamp::now();
//     let result: cudaError_t = unsafe { cudaGetLastError() };
//     let end = network::NsTimestamp::now();
//     let elapsed = (end.sec_timestamp - start.sec_timestamp) * 1000000000
//                 + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as i64;
//     info!("{}", elapsed);

//     let start = network::NsTimestamp::now();
//     result.send(channel_sender).unwrap();
//     channel_sender.flush_out().unwrap();
//     let end = network::NsTimestamp::now();
//     let elapsed = (end.sec_timestamp - start.sec_timestamp) * 1000000000
//                 + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as i64;
//     log::info!("server flushout end, {}:{} ", end.sec_timestamp, end.ns_timestamp);
//     info!("{}", elapsed);
// }
gen_exe!("cudaPeekAtLastError", "cudaError_t");
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
