#![allow(non_snake_case)]
use super::*;
use cudasys::types::cudart::*;

#[cfg(feature = "local")]
gen_hijack_local!(
    0,
    "cudaGetDevice",
    "cudaError_t",
    "*mut ::std::os::raw::c_int"
);
#[cfg(not(feature = "local"))]
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
// #[no_mangle]
// pub extern "C" fn cudaGetLastError() -> cudaError_t {
//     assert_eq!(true, *ENABLE_LOG);
//     info!("[{}:{}] cudaGetLastError", std::file!(), std::line!());
//     let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
//     let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
//     let proc_id = 3;
//     let mut result: cudaError_t = Default::default();
//     let start = network::NsTimestamp::now();
//     match proc_id.send(channel_sender) {
//         Ok(()) => {}
//         Err(e) => panic!("failed to send proc_id: {:?}", e),
//     }
//     match channel_sender.flush_out() {
//         Ok(()) => {}
//         Err(e) => panic!("failed to send: {:?}", e),
//     }
//     let end = network::NsTimestamp::now();
//     let elapsed = (end.sec_timestamp - start.sec_timestamp) * 1000000000
//                 + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as i64;
//     info!("{}", elapsed);

//     let start = network::NsTimestamp::now();
//     match result.recv(channel_receiver) {
//         Ok(()) => {}
//         Err(e) => panic!("failed to receive result: {:?}", e),
//     }
//     let end = network::NsTimestamp::now();
//     let elapsed = (end.sec_timestamp - start.sec_timestamp) * 1000000000
//                 + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as i64;
//     info!("{}", elapsed);

//     let start = network::NsTimestamp::now();
//     match channel_receiver.recv_ts() {
//         Ok(()) => {}
//         Err(e) => panic!("failed to receive timestamp: {:?}", e),
//     }
//     let end = network::NsTimestamp::now();
//     let elapsed = (end.sec_timestamp - start.sec_timestamp) * 1000000000
//                 + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as i64;
//     info!("{}", elapsed);
//     return result;
// }
gen_hijack!(4, "cudaPeekAtLastError", "cudaError_t");
gen_hijack!(5, "cudaStreamSynchronize", "cudaError_t", "cudaStream_t");
gen_hijack!(6, "cudaMalloc", "cudaError_t", "*mut MemPtr", "size_t");
// gen_hijack!(
//     7,
//     "cudaMemcpy",
//     "cudaError_t",
//     "MemPtr",
//     "MemPtr",
//     "size_t",
//     "cudaMemcpyKind"
// );
#[cfg(feature = "async_api")]
gen_hijack_async!(8, "cudaFree", "cudaError_t", "MemPtr");
#[cfg(not(feature = "async_api"))]
gen_hijack!(8, "cudaFree", "cudaError_t", "MemPtr");
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
gen_hijack!(
    12,
    "cudaPointerGetAttributes", 
    "cudaError_t", 
    "*mut cudaPointerAttributes", 
    "MemPtr"
);
gen_hijack!(
    14,
    "cudaFuncGetAttributes", 
    "cudaError_t", 
    "*mut cudaFuncAttributes", 
    "MemPtr"
);
// gen_hijack!(
//     100,
//     "__cudaRegisterFatBinary",
//     "MemPtr",
//     "*const ::std::os::raw::c_void"
// );
// gen_hijack!(
//     101,
//     "__cudaUnregisterFatBinary",
//     "null",
//     "MemPtr"
// );
// gen_hijack!(
//     102,
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
// gen_hijack!(
//     103,
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
// gen_hijack!(
//     200,
//     "cudaLaunchKernel",
//     "cudaError_t",
//     "*const ::std::os::raw::c_void",
//     "dim3",
//     "dim3",
//     "*mut *mut ::std::os::raw::c_void",
//     "usize",
//     "cudaStream_t"
// );
