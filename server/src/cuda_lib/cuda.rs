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
