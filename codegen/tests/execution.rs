#![allow(non_snake_case)]

extern crate codegen;

use codegen::gen_exe;
use network::{cudaError_t, CommChannel};
use log::info;

extern "C" {
    pub fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
}

gen_exe!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
gen_exe!("cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
