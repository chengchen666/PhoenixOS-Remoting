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

extern "C" {
    pub fn cudaGetLastError() -> cudaError_t;
}

extern "C" {
    pub fn cudaPeekAtLastError() -> cudaError_t;
}

extern "C" {
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}

extern "C" {
    pub fn cudaMalloc(devPtr: *mut MemPtr, size: usize) -> cudaError_t;
}

extern "C" {
    pub fn cudaMemcpy(dst: MemPtr, src: MemPtr, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
}

extern "C" {
    pub fn cudaFree(devPtr: MemPtr) -> cudaError_t;
}

extern "C" {
    pub fn cudaStreamIsCapturing(
        stream: cudaStream_t,
        pCaptureStatus: *mut cudaStreamCaptureStatus,
    ) -> cudaError_t;
}

extern "C" {
    pub fn cudaGetDeviceProperties(
        prop: *mut cudaDeviceProp,
        device: ::std::os::raw::c_int,
    ) -> cudaError_t;
}

extern "C" {
    pub fn cudaLaunchKernel(
        func: CUfunction,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut ::std::os::raw::c_void,
        sharedMem: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn cuda_ffi() {
        let mut device = 0;
        let mut device_num = 0;

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDeviceCount(&mut device_num as *mut i32) }
        {
            println!("device count: {}", device_num);
        } else {
            panic!("failed to get device count");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, 0);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaSetDevice(device_num - 1) } {
        } else {
            panic!("failed to set device");
        }

        if let cudaError_t::cudaSuccess = unsafe { cudaGetDevice(&mut device as *mut i32) } {
            assert_eq!(device, device_num - 1);
            println!("device: {}", device);
        } else {
            panic!("failed to get device");
        }
    }
}
