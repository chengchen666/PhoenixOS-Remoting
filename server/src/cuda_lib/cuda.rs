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
