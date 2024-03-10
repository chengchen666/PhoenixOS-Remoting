use super::*;

extern "C" {
    pub fn nvmlInit_v2() -> nvmlReturn_t;
}

extern "C" {
    pub fn nvmlDeviceGetCount_v2(deviceCount: *mut ::std::os::raw::c_uint) -> nvmlReturn_t;
}

extern "C" {
    pub fn nvmlInitWithFlags(flags: ::std::os::raw::c_uint) -> nvmlReturn_t;
}
