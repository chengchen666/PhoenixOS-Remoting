use codegen::cuda_hook;
use crate::types::nvml::*;
use std::os::raw::*;

#[cuda_hook(proc_id = 1000)]
fn nvmlInit_v2() -> nvmlReturn_t;

#[cuda_hook(proc_id = 1001)]
fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 1002)]
fn nvmlInitWithFlags(flags: c_uint) -> nvmlReturn_t;
