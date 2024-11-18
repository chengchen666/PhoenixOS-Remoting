use crate::types::cuda::*;
use codegen::cuda_hook;
use std::os::raw::*;

#[cuda_hook(proc_id = 300)]
fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut c_uint, active: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 501)]
fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;

#[cuda_hook(proc_id = 502)]
fn cuInit(Flags: c_uint) -> CUresult;

#[cuda_hook(proc_id = 504)]
fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
