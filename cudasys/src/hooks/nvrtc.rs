use crate::types::nvrtc::*;
use codegen::cuda_hook;
use std::os::raw::*;

#[cuda_hook(proc_id = 3000)]
fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult;
