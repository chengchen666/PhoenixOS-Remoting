use super::*;

extern "C" {
    pub fn cudaDeviceSynchronize() -> cudaError_t;
}

extern "C" {
    pub fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
}

extern "C" {
    pub fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
}

extern "C" {
    pub fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t;
}
