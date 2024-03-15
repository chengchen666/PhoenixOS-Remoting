use super::*;

extern "C" {
    pub fn cuModuleLoadData(
        module: *mut CUmodule,
        image: *const ::std::os::raw::c_void,
    ) -> CUresult;
}

extern "C" {
    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;
}

extern "C" {
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const ::std::os::raw::c_char,
    ) -> CUresult;
}

extern "C" {
    pub fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        hmod: CUmodule,
        name: *const ::std::os::raw::c_char,
    ) -> CUresult;
}

extern "C" {
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: ::std::os::raw::c_uint,
        gridDimY: ::std::os::raw::c_uint,
        gridDimZ: ::std::os::raw::c_uint,
        blockDimX: ::std::os::raw::c_uint,
        blockDimY: ::std::os::raw::c_uint,
        blockDimZ: ::std::os::raw::c_uint,
        sharedMemBytes: ::std::os::raw::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::std::os::raw::c_void,
        extra: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;
}

extern "C" {
    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut ::std::os::raw::c_uint,
        active: *mut ::std::os::raw::c_int,
    ) -> CUresult;
}
