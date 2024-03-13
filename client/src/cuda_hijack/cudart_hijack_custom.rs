#![allow(non_snake_case)]
use super::*;

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    println!("[{}:{}] cudaMemcpy", std::file!(), std::line!());
    let channel_sender = &mut (*CHANNEL_SENDER.lock().unwrap());
    let channel_receiver = &mut (*CHANNEL_RECEIVER.lock().unwrap());

    if cudaMemcpyKind::cudaMemcpyHostToHost == kind {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
        }
        return cudaError_t::cudaSuccess;
    }

    let proc_id = 7;
    let mut result: cudaError_t = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match dst.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send dst: {:?}", e),
    }
    match src.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send src: {:?}", e),
    }
    match count.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send count: {:?}", e),
    }
    match kind.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send kind: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        // transport [src; count] to device
        let data = unsafe { std::slice::from_raw_parts(src as *const u8, count) };
        match data.send(channel_sender) {
            Ok(()) => {}
            Err(e) => panic!("failed to send data: {:?}", e),
        }
    }

    match channel_sender.flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        // receive [dst; count] from device
        let data = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, count) };
        match data.recv(channel_receiver) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive data: {:?}", e),
        }
    }

    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    return result;
}

// TODO: maybe we should understand the semantic diff of cudaMemcpyAsync&cudaMemcpy
#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
    _stream: cudaStream_t,
) -> cudaError_t {
    println!("[{}:{}] cudaMemcpyAsync", std::file!(), std::line!());
    cudaMemcpy(dst, src, count, kind)
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinary(fatCubin: *const ::std::os::raw::c_void) -> MemPtr {
    println!(
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
    return client_address;
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(fatCubinHandle: MemPtr) {
    println!(
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
    if CUresult::CUDA_SUCCESS != result {
        panic!("error unregistering fatbin: {:?}", result);
    }
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(fatCubinHandle: MemPtr) {
    println!(
        "[{}:{}] __cudaRegisterFatBinaryEnd",
        std::file!(),
        std::line!()
    );
}
