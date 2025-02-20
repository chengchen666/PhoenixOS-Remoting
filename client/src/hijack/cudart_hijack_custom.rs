#![expect(non_snake_case)]
use super::*;
use cudasys::types::cudart::*;
use std::cell::RefCell;
use std::ffi::*;
use std::alloc::{alloc, Layout};

#[no_mangle]
#[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
pub extern "C" fn cudaMemcpy(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    log::debug!("[{}:{}] cudaMemcpy", std::file!(), std::line!());

    assert_ne!(kind, cudaMemcpyKind::cudaMemcpyDefault, "cudaMemcpyDefault is not supported yet");

    let ClientThread { channel_sender, channel_receiver, .. } = client;

    if cudaMemcpyKind::cudaMemcpyHostToHost == kind {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
        }
        return cudaError_t::cudaSuccess;
    }

    let proc_id = 278;
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
        #[cfg(feature = "async_api")]
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
    }

    if cfg!(feature = "async_api") {
        return cudaError_t::cudaSuccess;
    } else {
        match result.recv(channel_receiver) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive result: {:?}", e),
        }
        match channel_receiver.recv_ts() {
            Ok(()) => {}
            Err(e) => panic!("failed to receive timestamp: {:?}", e),
        }
        return result;
    }
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
    log::debug!("[{}:{}] cudaMemcpyAsync", std::file!(), std::line!());
    cudaMemcpy(dst, src, count, kind)
}

#[no_mangle]
pub extern "C" fn cudaLaunchKernel(
    func: MemPtr,
    gridDim: dim3,
    blockDim: dim3,
    args: *mut *mut ::std::os::raw::c_void,
    sharedMem: usize,
    stream: cudaStream_t,
) -> cudaError_t {
    log::debug!("[{}:{}] cudaLaunchKernel", std::file!(), std::line!());

    use super::cuda_hijack::{cuLaunchKernel, cuModuleGetFunction, cuModuleLoadData};

    let cufunc = RUNTIME_CACHE.with_borrow_mut(|runtime| {
        if runtime.loaded_modules.is_empty() {
            log::info!(
                "#fatbins = {}, #functions = {}",
                runtime.lazy_fatbins.len(),
                runtime.lazy_functions.len(),
            );
        }
        let load_module = |fatCubinHandle: &FatBinaryHandle| {
            // See our implementation of `__cudaRegisterFatBinary`
            let index = (*fatCubinHandle >> 4) - 1;
            log::info!("registering fatbin #{index}");
            let image = runtime.lazy_fatbins[index];
            CLIENT_THREAD.with_borrow_mut(|client| client.is_cuda_launch_kernel = true);
            let mut module = std::ptr::null_mut();
            assert_eq!(cuModuleLoadData(&raw mut module, image.cast()), Default::default());
            CLIENT_THREAD.with_borrow_mut(|client| client.is_cuda_launch_kernel = false);
            module
        };
        let load_function = |func: &HostPtr| {
            let (fatCubinHandle, deviceName) = *runtime.lazy_functions.get(func).unwrap();
            log::info!("registering function {:?}", unsafe { CStr::from_ptr(deviceName) });
            let module =
                *runtime.loaded_modules.entry(fatCubinHandle).or_insert_with_key(load_module);
            let mut cufunc = std::ptr::null_mut();
            assert_eq!(
                cuModuleGetFunction(&raw mut cufunc, module, deviceName),
                Default::default()
            );
            cufunc
        };
        *runtime.loaded_functions.entry(func).or_insert_with_key(load_function)
    });

    unsafe {
        std::mem::transmute(cuLaunchKernel(
            cufunc,
            gridDim.x,
            gridDim.y,
            gridDim.z,
            blockDim.x,
            blockDim.y,
            blockDim.z,
            sharedMem.try_into().unwrap(),
            stream.cast(),
            args,
            std::ptr::null_mut(),
        ))
    }
}

#[no_mangle]
pub extern "C" fn cudaHostAlloc(
    pHost: *mut *mut ::std::os::raw::c_void,
    size: usize,
    flags: c_uint,
) -> cudaError_t {
    log::debug!(
        "[{}:{}] cudaHostAlloc",
        std::file!(),
        std::line!()
    );
    assert_eq!(flags, cudaHostAllocDefault);
    let Ok(layout) = Layout::from_size_align(size as _, 1) else {
        return cudaError_t::cudaErrorInvalidValue;
    };
    // TODO: handle pinned memory at server side in a better way
    unsafe {
        let ptr = alloc(layout);
        if ptr.is_null() {
            return cudaError_t::cudaErrorMemoryAllocation;
        }
        *pHost = ptr as _;
    }
    cudaError_t::cudaSuccess
}

#[no_mangle]
#[use_thread_local(client = CLIENT_THREAD.with_borrow_mut)]
pub extern "C" fn cudaGetErrorString(
    cudaError: cudaError_t,
) -> *const ::std::os::raw::c_char {
    log::debug!(
        "[{}:{}] cudaGetErrorString",
        std::file!(),
        std::line!()
    );
    let ClientThread { channel_sender, channel_receiver, .. } = client;
    let proc_id = 151;
    let mut result:Vec<u8>  = Default::default();
    match proc_id.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match cudaError.send(channel_sender) {
        Ok(()) => {}
        Err(e) => panic!("failed to send cudaError: {:?}", e),
    }
    channel_sender.flush_out().unwrap();
    match result.recv(channel_receiver) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    match channel_receiver.recv_ts() {
                Ok(()) => {}
                Err(e) => panic!("failed to receive timestamp: {:?}", e),
            }
    let result = CString::new(result).unwrap();
    result.into_raw() // leaking the string as the program is about to fail anyway
}

struct CallConfiguration {
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: MemPtr,
}

thread_local! {
    static CALL_CONFIGURATIONS: RefCell<Vec<CallConfiguration>> = const { RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn __cudaPushCallConfiguration(
    gridDim: dim3,
    blockDim: dim3,
    sharedMem: usize,
    stream: MemPtr,
) -> cudaError_t {
    CALL_CONFIGURATIONS.with_borrow_mut(|v| {
        v.push(CallConfiguration {
            gridDim,
            blockDim,
            sharedMem,
            stream,
        });
    });
    cudaError_t::cudaSuccess
}

#[no_mangle]
pub extern "C" fn __cudaPopCallConfiguration(
    gridDim: *mut dim3,
    blockDim: *mut dim3,
    sharedMem: *mut usize,
    stream: *mut MemPtr,
) -> cudaError_t {
    if let Some(config) = CALL_CONFIGURATIONS.with_borrow_mut(Vec::pop) {
        unsafe {
            *gridDim = config.gridDim;
            *blockDim = config.blockDim;
            *sharedMem = config.sharedMem;
            *stream = config.stream;
        }
        cudaError_t::cudaSuccess
    } else {
        cudaError_t::cudaErrorMissingConfiguration
    }
}
