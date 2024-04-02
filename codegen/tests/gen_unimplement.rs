#![allow(non_snake_case)]
extern crate lazy_static;

extern crate cudasys;
extern crate network;

use cudasys::cuda::{CUmodule, CUjit_option, CUresult};

extern crate codegen;

use codegen::gen_unimplement;

gen_unimplement!("pub fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const ::std::os::raw::c_void,
    numOptions: ::std::os::raw::c_uint,
    options: *mut CUjit_option,
    optionValues: *mut *mut ::std::os::raw::c_void,
) -> CUresult;");
gen_unimplement!("pub fn cuModuleLoadFatBinary(
    module: *mut CUmodule,
    fatCubin: *const ::std::os::raw::c_void,
) -> CUresult;");
