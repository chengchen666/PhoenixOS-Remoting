#![expect(non_snake_case)]
use super::*;

// original dlsym
extern "C" {
    pub fn dlsym(handle: *mut std::ffi::c_void, symbol: *const std::os::raw::c_char) -> *mut std::ffi::c_void;
}

// dlopen_orig and dlclose_orig are used to load the original dlopen and dlsym functions from the libdl
lazy_static! {
    // # define RTLD_NEXT        ((void *) -1l)
    static ref DLOPEN_ORIG: extern "C" fn(*const std::os::raw::c_char, std::os::raw::c_int) -> *mut std::ffi::c_void = {
        let RTLD_NEXT = usize::MAX as *mut std::ffi::c_void;
        let symbol = b"dlopen\0".as_ptr() as *const std::os::raw::c_char;
        let orig = unsafe { dlsym(RTLD_NEXT, symbol) };
        if orig.is_null() {
            panic!("Failed to load original dlopen");
        }
        unsafe { std::mem::transmute(orig) }
    };
    static ref DLCLOSE_ORIG: extern "C" fn(*mut std::ffi::c_void) -> std::os::raw::c_int = {
        let RTLD_NEXT = usize::MAX as *mut std::ffi::c_void;
        let symbol = b"dlclose\0".as_ptr() as *const std::os::raw::c_char;
        let orig = unsafe { dlsym(RTLD_NEXT, symbol) };
        if orig.is_null() {
            panic!("Failed to load original dlclose");
        }
        unsafe { std::mem::transmute(orig) }
    };
    static ref SELF_HANDLES: Mutex<Vec<usize>> = Mutex::new(Vec::new());
}

#[no_mangle]
pub extern "C" fn dlopen(filename: *const std::os::raw::c_char, flags: std::os::raw::c_int) -> *mut std::ffi::c_void {
    // use the original dlopen to load the library
    if filename.is_null() {
        return DLOPEN_ORIG(filename, flags);
    }
    let filename = unsafe { std::ffi::CStr::from_ptr(filename) };
    let filename = filename.to_str().unwrap();
    if filename.contains("libcuda") || filename.contains("libnvrtc") || filename.contains("libnvidia-ml") {
        // if the library is libcuda, libnvrtc or libnvidia-ml, return a handle to the client
        info!("[dlopen] replacing dlopen call to {} library with a handle to the client", filename);
        let self_handle = DLOPEN_ORIG("libclient.so\0".as_ptr() as *const std::os::raw::c_char, flags);
        if self_handle.is_null() {
            panic!("Failed to load the client handle");
        }
        let mut self_handles = SELF_HANDLES.lock().unwrap();
        self_handles.push(self_handle as usize);
        return self_handle;
    }
    return DLOPEN_ORIG(filename.as_ptr() as *const std::os::raw::c_char, flags);
}

#[no_mangle]
pub extern "C" fn dlclose(handle: *mut std::ffi::c_void) -> std::os::raw::c_int {
    let self_handles = SELF_HANDLES.lock().unwrap();
    if self_handles.contains(&(handle as usize)) {
        // if the handle is the client handle, return 0
        info!("[dlclose] ignoring dlclose call to the client handle");
        return 0;
    }
    return DLCLOSE_ORIG(handle);
}
