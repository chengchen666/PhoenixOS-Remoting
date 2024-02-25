#![allow(non_snake_case)]
use super::*;

gen_hijack!(0, "cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_hijack!(1, "cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_hijack!(2, "cudaGetDeviceCount", "cudaError_t", "*mut ::std::os::raw::c_int");

// #[no_mangle]
// pub extern "C" fn cudaGetDevice(param1: *mut ::std::os::raw::c_int) -> cudaError_t {
//     println!("[{}:{}] cudaGetDevice", std::file!(), std::line!());
//     let proc_id = 0;
//     let mut var1 = Default::default();
//     let mut result = Default::default();

//     match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to serialize proc_id: {:?}", e),
//     }
//     match CHANNEL_SENDER.lock().unwrap().flush_out() {
//         Ok(_) => {}
//         Err(e) => panic!("failed to send: {:?}", e),
//     }

//     match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut var1) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to deserialize var1: {:?}", e),
//     }
//     match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to deserialize result: {:?}", e),
//     }
//     unsafe {
//         *param1 = var1;
//     }
//     return result;
// }

// #[no_mangle]
// pub extern "C" fn cudaSetDevice(param1: ::std::os::raw::c_int) -> cudaError_t {
//     println!("[{}:{}] cudaSetDevice", std::file!(), std::line!());
//     let proc_id = 1;
//     let mut result = Default::default();
//     match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to serialize proc_id: {:?}", e),
//     }
//     match CHANNEL_SENDER.lock().unwrap().send_var(&param1) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to serialize param1: {:?}", e),
//     }
//     match CHANNEL_SENDER.lock().unwrap().flush_out() {
//         Ok(_) => {}
//         Err(e) => panic!("failed to send: {:?}", e),
//     }

//     match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to deserialize result: {:?}", e),
//     }
//     return result;
// }

// #[no_mangle]
// pub extern "C" fn cudaGetDeviceCount(param1: *mut ::std::os::raw::c_int) -> cudaError_t {
//     println!("[{}:{}] cudaGetDeviceCount", std::file!(), std::line!());
//     let proc_id = 2;
//     let mut var1 = Default::default();
//     let mut result = Default::default();
//     match CHANNEL_SENDER.lock().unwrap().send_var(&proc_id) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to serialize proc_id: {:?}", e),
//     }
//     match CHANNEL_SENDER.lock().unwrap().flush_out() {
//         Ok(_) => {}
//         Err(e) => panic!("failed to send: {:?}", e),
//     }

//     match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut var1) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to deserialize var1: {:?}", e),
//     }
//     match CHANNEL_RECEIVER.lock().unwrap().recv_var(&mut result) {
//         Ok(_) => {}
//         Err(e) => panic!("failed to deserialize result: {:?}", e),
//     }
//     unsafe {
//         *param1 = var1;
//     }
//     return result;
// }
