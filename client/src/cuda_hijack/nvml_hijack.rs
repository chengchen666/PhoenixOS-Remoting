#![allow(non_snake_case)]
use super::*;

gen_hijack!(1000, "nvmlInit_v2", "nvmlReturn_t");
gen_hijack!(
    1001,
    "nvmlDeviceGetCount_v2",
    "nvmlReturn_t",
    "*mut ::std::os::raw::c_uint"
);
gen_hijack!(
    1002,
    "nvmlInitWithFlags",
    "nvmlReturn_t",
    "::std::os::raw::c_uint"
);
