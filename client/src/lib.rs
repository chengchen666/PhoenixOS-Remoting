pub mod cuda_runtime;
pub use cuda_runtime::cudaError_t;

#[no_mangle]
pub extern "C" fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t {
    unsafe {
        *device = 0;
    }
    println!("cudaGetDevice");
    return cudaError_t::cudaSuccess;
}

#[no_mangle]
pub extern "C" fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t {
    println!("cudaSetDevice: {}", device);
    return cudaError_t::cudaSuccess;
}
