#![allow(non_snake_case)]
extern crate lazy_static;

extern crate cudasys;
extern crate network;

use cudasys::cudart::cudaError_t;

extern crate codegen;

use codegen::gen_unimplement;

gen_unimplement!("cudaGetDevice", "cudaError_t", "*mut ::std::os::raw::c_int");
gen_unimplement!("cudaSetDevice", "cudaError_t", "::std::os::raw::c_int");
