#![allow(non_snake_case)]
use super::*;
use cudasys::types::cuda::*;
use std::ffi::CStr;

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinary(fatCubin: *const ::std::os::raw::c_void) -> MemPtr {
    assert_eq!(true, *ENABLE_LOG);
    info!(
        "[{}:{}] __cudaRegisterFatBinary",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    let proc_id = 100;
    let mut result: CUresult = Default::default();

    let mut fatbin_ptr: *mut u8 = std::ptr::null_mut();
    let mut fatbin_size: usize = Default::default();
    if 0 != ELF_CONTROLLER.elf2_get_fatbin_info(
        fatCubin as *const fat_header,
        &mut fatbin_ptr as *mut *mut u8,
        &mut fatbin_size as *mut usize,
    ) {
        panic!("error getting fatbin info");
    }
    let fatbin: Vec<u8> = unsafe { std::slice::from_raw_parts(fatbin_ptr, fatbin_size).to_vec() };

    // CUDA registers an atexit handler for fatbin cleanup that accesses
    // the fatbin data structure. Let's allocate some zeroes to avoid segfaults.
    let client_address: MemPtr = unsafe {
        std::alloc::alloc(std::alloc::Layout::from_size_align(0x58, 8).unwrap()) as MemPtr
    };

    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match fatbin.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send fatbin: {:?}", e),
    }
    match client_address.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send client_address: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    if CUresult::CUDA_SUCCESS != result {
        panic!("error registering fatbin: {:?}", result);
    }
    info!("after recv");
    match channel_receiver.recv_ts() {
        Ok(()) => {}
        Err(e) => panic!("failed to receive timestamp: {:?}", e),
    }
    info!("after recv");
    return client_address;
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle: MemPtr) {
    assert_eq!(true, *ENABLE_LOG);
    info!(
        "[{}:{}] __cudaUnregisterFatBinary",
        std::file!(),
        std::line!()
    );
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    let proc_id = 101;
    let mut result: CUresult = Default::default();

    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match fatCubinHandle.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send fatCubinHandle: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if CUresult::CUDA_SUCCESS != result {
        panic!("error unregistering fatbin: {:?}", result);
    }
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(_fatCubinHandle: MemPtr) {
    assert_eq!(true, *ENABLE_LOG);
    info!(
        "[{}:{}] __cudaRegisterFatBinaryEnd",
        std::file!(),
        std::line!()
    );
    // TODO: no actual impact
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFunction(
    fatCubinHandle: MemPtr,
    hostFun: MemPtr,
    _deviceFun: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    _thread_limit: ::std::os::raw::c_int,
    _tid: MemPtr,
    _bid: MemPtr,
    _bDim: MemPtr,
    _gDim: MemPtr,
    _wSize: MemPtr,
) {
    assert_eq!(true, *ENABLE_LOG);
    info!("[{}:{}] __cudaRegisterFunction", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    let proc_id = 102;
    let mut result: CUresult = Default::default();

    let info = ELF_CONTROLLER.utils_search_info(deviceName);
    if info.is_null() {
        panic!("request to register unknown function: {:?}", deviceName);
    }
    let info = unsafe { &mut *info };
    let mut deviceName: Vec<u8> = unsafe { std::ffi::CStr::from_ptr(deviceName) }
        .to_bytes()
        .to_vec();
    // append a null terminator!!!
    deviceName.push(0);

    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match fatCubinHandle.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send fatCubinHandle: {:?}", e),
    }
    match hostFun.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send hostFun: {:?}", e),
    }
    match deviceName.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send deviceName: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if CUresult::CUDA_SUCCESS != result {
        let c_str = unsafe{ CStr::from_ptr(_deviceFun) };
        match c_str.to_str() {
            Ok(rust_str) => println!("function name: {}", rust_str),
            Err(_) => eprintln!("Failed to convert C string to Rust string"),
        }
        panic!("error registering function: {:?}", result);
    }

    info.host_fun = hostFun as *mut ::std::os::raw::c_void;
    ELF_CONTROLLER.add_kernel_host_func(hostFun as *mut ::std::os::raw::c_void, info);
}

#[no_mangle]
pub extern "C" fn __cudaRegisterVar(
    fatCubinHandle: MemPtr,
    hostVar: MemPtr,
    _deviceAddress: *mut ::std::os::raw::c_char,
    deviceName: *const ::std::os::raw::c_char,
    _ext: ::std::os::raw::c_int,
    _size: usize,
    _constant: ::std::os::raw::c_int,
    _global: ::std::os::raw::c_int,
) {
    assert_eq!(true, *ENABLE_LOG);
    info!("[{}:{}] __cudaRegisterVar", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    let proc_id = 103;
    let mut result: CUresult = Default::default();

    let mut deviceName: Vec<u8> = unsafe { std::ffi::CStr::from_ptr(deviceName) }
        .to_bytes()
        .to_vec();
    deviceName.push(0);

    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match fatCubinHandle.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send fatCubinHandle: {:?}", e),
    }
    match hostVar.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send hostVar: {:?}", e),
    }
    match deviceName.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send deviceName: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if CUresult::CUDA_SUCCESS != result {
        panic!("error registering var: {:?}", result);
    }
}


#[no_mangle]
pub extern "C" fn cuGetProcAddress(
   symbol: *const ::std::os::raw::c_char,
   pfn: MemPtr,
   cudaVersion: ::std::os::raw::c_int,
   flags: cuuint64_t, 
){
    info!("[{}:{}] cuGetProcAddress", std::file!(), std::line!());
    let proc_id = 500;
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let mut result: CUresult = Default::default();
    let mut symbol_: Vec<u8> = unsafe { std::ffi::CStr::from_ptr(symbol) }
        .to_bytes()
        .to_vec();
    symbol_.push(0);
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match symbol_.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send symbol: {:?}", e),
    }
    match cudaVersion.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send cudaVersion: {:?}", e),
    }
    match flags.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send flags: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    
    let mut func_ptr: MemPtr = 0; // *void
    match func_ptr.recv(channel_receiver) {
        Ok(()) => {
           unsafe{
                *(pfn as *mut *mut ::std::os::raw::c_void) = func_ptr as *mut ::std::os::raw::c_void
            };
        }
        Err(e) => panic!("failed to receive pfn: {:?}", e),
    }
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if CUresult::CUDA_SUCCESS != result {
        panic!("error getting function address: {:?}", result);
    }
}


#[no_mangle]
pub extern "C" fn cuGetExportTable(
    ppExportTable: MemPtr,
    pExportTableId: MemPtr,
) {
    info!("[{}:{}] cuGetExportTable", std::file!(), std::line!());
    let proc_id = 503; 
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());
    let mut result: CUresult = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    let pExportTableId_: CUuuid = unsafe { *(pExportTableId as *const CUuuid) };
    match pExportTableId_.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send pExportTableId: {:?}", e),
    }
    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }
    let mut ppExportTable_: MemPtr = 0; // *const ::std::os::raw::c_void
    match ppExportTable_.recv(channel_receiver) {
        Ok(()) => {
            unsafe{
                *(ppExportTable as *mut *const ::std::os::raw::c_void) = ppExportTable_ as *const ::std::os::raw::c_void
            };
        }
        Err(e) => panic!("failed to receive ppExportTable: {:?}", e),
    }
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    if CUresult::CUDA_SUCCESS != result {
        panic!("error getting export table: {:?}", result);
    }
}