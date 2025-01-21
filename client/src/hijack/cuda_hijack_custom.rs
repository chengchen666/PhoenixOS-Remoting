#![expect(non_snake_case)]
use super::*;

use std::cell::OnceCell;

#[no_mangle]
extern "C" fn __cudaRegisterFatBinary(fatCubin: *const FatBinaryWrapper) -> FatBinaryHandle {
    let fatCubin = unsafe { (*fatCubin).unwrap() };

    #[thread_local]
    static VALIDATE: OnceCell<bool> = OnceCell::new();

    if *VALIDATE.get_or_init(|| std::env::var_os("VALIDATE_FAT_BINARY").is_some()) {
        unsafe { (*fatCubin).validate_code() };
    }

    RUNTIME_CACHE.with_borrow_mut(|runtime| {
        runtime.lazy_fatbins.push(fatCubin);
        runtime.lazy_fatbins.len() << 4
    })
}

#[no_mangle]
pub extern "C" fn __cudaUnregisterFatBinary(_fatCubinHandle: MemPtr) {
    // This is called when the client process exits, when the thread local storage is already dropped.
    // Therefore, we implement this at the server side.
}

#[no_mangle]
pub extern "C" fn __cudaRegisterFatBinaryEnd(_fatCubinHandle: MemPtr) {
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
    if cfg!(debug_assertions) && deviceName != _deviceFun {
        log::warn!(
            "deviceName: {:?}, deviceFun: {:?}",
            unsafe { std::ffi::CStr::from_ptr(deviceName) },
            unsafe { std::ffi::CStr::from_ptr(_deviceFun) },
        );
    }

    // Some kernels are registered multiple times from different fatbins
    // e.g. "void cub::EmptyKernel<void>()"
    RUNTIME_CACHE.with_borrow_mut(|runtime| {
        runtime.lazy_functions.entry(hostFun).or_insert((fatCubinHandle, deviceName));
    });
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
    if cfg!(debug_assertions) && deviceName != _deviceAddress {
        log::warn!(
            "deviceName: {:?}, deviceFun: {:?}",
            unsafe { std::ffi::CStr::from_ptr(deviceName) },
            unsafe { std::ffi::CStr::from_ptr(_deviceAddress) },
        );
    }

    assert!(RUNTIME_CACHE.with_borrow_mut(|runtime| {
        runtime.lazy_variables.insert(hostVar, (fatCubinHandle, deviceName)).is_none()
    }));
}
