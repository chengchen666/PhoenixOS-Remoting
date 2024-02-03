use super::*;

extern "C" {
    pub fn cudaDeviceSynchronize() -> cudaError_t;
}

extern "C" {
    pub fn cudaSetDevice(device: ::std::os::raw::c_int) -> cudaError_t;
}

extern "C" {
    pub fn cudaGetDevice(device: *mut ::std::os::raw::c_int) -> cudaError_t;
}

extern "C" {
    pub fn cudaGetDeviceCount(count: *mut ::std::os::raw::c_int) -> cudaError_t;
}

#[cfg(test)]
mod tests{
    use super::*;
    use num::FromPrimitive;

    #[test]
    fn test_num_derive() {
        let x: u32 = cudaError_t::cudaSuccess as u32;
        assert_eq!(x, 0);
        match cudaError_t::from_u32(1) {
            Some(v) => assert_eq!(v, cudaError_t::cudaErrorInvalidValue),
            None => panic!("failed to convert from u32"),
        }
    }
}
