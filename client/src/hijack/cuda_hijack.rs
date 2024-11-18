use super::*;
use cudasys::types::cuda::*;
use std::os::raw::*;

#[cuda_hook_hijack(proc_id = 300)]
fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult;

#[cuda_hook_hijack(proc_id = 501)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_hook_hijack(proc_id = 502)]
fn cuInit(Flags: c_uint) -> CUresult;

#[cuda_hook_hijack(proc_id = 504)]
fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
