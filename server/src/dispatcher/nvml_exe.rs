#![allow(non_snake_case)]
use super::*;
use cudasys::nvml::*;

gen_exe!("nvmlInit_v2", "nvmlReturn_t");
gen_exe!("nvmlDeviceGetCount_v2", "nvmlReturn_t", "*mut ::std::os::raw::c_uint");
gen_exe!("nvmlInitWithFlags", "nvmlReturn_t", "::std::os::raw::c_uint");
