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

// extern "C" {
//     pub fn cudaMalloc(devPtr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;
// }

// extern "C" {
//     pub fn cudaMemcpy(
//         dst: *mut ::std::os::raw::c_void,
//         src: *const ::std::os::raw::c_void,
//         count: usize,
//         kind: cudaMemcpyKind,
//     ) -> cudaError_t;
// }

// extern "C" {
//     pub fn cudaMemcpyAsync(
//         dst: *mut ::std::os::raw::c_void,
//         src: *const ::std::os::raw::c_void,
//         count: usize,
//         kind: cudaMemcpyKind,
//         stream: cudaStream_t,
//     ) -> cudaError_t;
// }

// #[no_mangle]
// pub extern "C" fn __cudaRegisterFatBinary(fatCubin: &FatHeader) -> *mut u64 {
//     // println!("Hijacked __cudaRegisterFatBinary(fatCubin:{:#x?})", fatCubin);
//     let len = fatCubin.text.header_size as usize + fatCubin.text.size as usize;
//     let tempVaue = 0;
//     let result = &tempVaue as *const _ as u64;
//     unsafe {
//         syscall4(SYS_PROXY, ProxyCommand::CudaRegisterFatBinary as usize, len, fatCubin.text as *const _ as usize, result as usize);
//     }
//     return result as *mut u64;
// }

// #[no_mangle]
// pub extern "C" fn __cudaRegisterFatBinaryEnd(fatCubinHandle:u64) {
//     let fatCubinPtr: *const u64 = fatCubinHandle as *const u64;
//     // unsafe{
//     // println!("Hijacked __cudaUnregisterFatBinaryEnd( the content of fatCubinHandle = {:x})", *fatCubinPtr);
//     // }
// }

// #[no_mangle]
// pub extern "C" fn __cudaRegisterFunction(
//     fatCubinHandle:u64, 
//     hostFun:u64, 
//     deviceFun:u64, 
//     deviceName:u64, 
//     thread_limit:usize, 
//     tid:u64, 
//     bid:u64, 
//     bDim:u64, 
//     gDim:u64, 
//     wSize:usize
// ) {
//     // println!("Hijacked __cudaRegisterFunction(fatCubinHandle:{:x}, hostFun:{:x}, deviceFun:{:x}, deviceName:{:x}, thread_limit: {}, tid: {:x}, bid: {:x}, bDim: {:x}, gDim: {:x}, wSize: {})", fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);    
//     let info = RegisterFunctionInfo {
//         fatCubinHandle: fatCubinHandle, 
//         hostFun: hostFun, 
//         deviceFun: deviceFun, 
//         deviceName: deviceName, 
//         thread_limit: thread_limit, 
//         tid: tid, 
//         bid: bid, 
//         bDim: bDim, 
//         gDim: gDim, 
//         wSize: wSize
//     };
//     // println!("RegisterFunctionInfo {:x?}", info);
//     unsafe {
//         syscall2(SYS_PROXY, ProxyCommand::CudaRegisterFunction as usize, &info as *const _ as usize);
//     }
// }

// extern void CUDARTAPI __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
//     char *deviceAddress,
//     const char *deviceName, int ext,
//     size_t size, int constant, int global) {

//     }

// #[no_mangle]
// pub extern "C" fn cudaLaunchKernel(
//     func:u64, 
//     gridDim:Qdim3, 
//     blockDim:Qdim3, 
//     args:u64, 
//     sharedMem:usize, 
//     stream:u64
// ) {
//     // println!("Hijacked cudaLaunchKernel(func:{:x}, gridDim:{:x?}, blockDim:{:x?}, args:{:x}, sharedMem: {}, stream: {:x?})", 
//     //    func, gridDim, blockDim, args, sharedMem, stream);
//     let info = LaunchKernelInfo {
//         func: func, 
//         gridDim: gridDim, 
//         blockDim: blockDim, 
//         args: args, 
//         sharedMem: sharedMem, 
//         stream: stream
//     };
//     unsafe {
//         syscall2(SYS_PROXY, ProxyCommand::CudaLaunchKernel as usize, &info as *const _ as usize);
//     }
// }

// extern "C" {
//     pub fn cudaLaunchKernel(
//         func: *const ::std::os::raw::c_void,
//         gridDim: dim3,
//         blockDim: dim3,
//         args: *mut *mut ::std::os::raw::c_void,
//         sharedMem: usize,
//         stream: cudaStream_t,
//     ) -> cudaError_t;
// }

// #[no_mangle]
// pub extern "C" fn cudaMalloc(
//         dev_ptr: *mut *mut c_void, 
//         size: usize
//     ) -> usize {
//     // println!("Hijacked cudaMalloc(size:{})", size);

//     let ret = unsafe {
//         syscall3(SYS_PROXY, ProxyCommand::CudaMalloc as usize, dev_ptr as * const _ as usize, size)
//     };
//     return ret;
// }

// extern "C" {
//     pub fn cudaMalloc(devPtr: *mut *mut ::std::os::raw::c_void, size: usize) -> cudaError_t;
// }

// #[no_mangle]
// pub extern "C" fn cudaMemcpy(
//         dst: *mut c_void, 
//         src: *const c_void, 
//         count: usize, 
//         kind: cudaMemcpyKind
//     ) -> usize {
//     // println!("Hijacked cudaMemcpy(size:{})", count);

//     if kind == cudaMemcpyKind::cudaMemcpyHostToHost {
//         unsafe {
//             std::ptr::copy_nonoverlapping(src as * const u8, dst as * mut u8, count);
//         }
        
//         return 0;
//     }

//     return unsafe {
//         syscall5(SYS_PROXY, ProxyCommand::CudaMemcpy as usize, dst as * const _ as usize, src as usize, count as usize, kind as usize) 
//     };
// }

// extern "C" {
//     pub fn cudaMemcpyAsync(
//         dst: *mut ::std::os::raw::c_void,
//         src: *const ::std::os::raw::c_void,
//         count: usize,
//         kind: cudaMemcpyKind,
//         stream: cudaStream_t,
//     ) -> cudaError_t;
// }

// extern "C" {
//     pub fn cudaStreamIsCapturing(
//         stream: cudaStream_t,
//         pCaptureStatus: *mut cudaStreamCaptureStatus,
//     ) -> cudaError_t;
// }

// extern "C" {
//     pub fn cudaGetDeviceProperties(
//         prop: *mut cudaDeviceProp,
//         device: ::std::os::raw::c_int,
//     ) -> cudaError_t;
// }


// #[no_mangle]
// pub extern "C" fn nvmlInitWithFlags(flags: ::std::os::raw::c_uint) -> nvmlReturn_t {
//     unsafe { crate::r#impl::init_with_flags(flags).into() }
// }

// #[no_mangle]
// pub extern "C" fn nvmlInit_v2() -> nvmlReturn_t {
//     unsafe { crate::r#impl::init().into() }
// }

// #[no_mangle]
// pub extern "C" fn nvmlDeviceGetCount_v2(deviceCount: *mut ::std::os::raw::c_uint) -> nvmlReturn_t {
//     unsafe { crate::r#impl::device_get_count(deviceCount) }.into()
// }


// https://github.com/QuarkContainer/Quark/blob/8c3f63eb6edcf1dd043e325512d47781951ff89a/cudaproxy/src/cudaproxy.rs#L75
// https://github.com/vosen/ZLUDA/blob/master/zluda_ml/src/nvml.rs#L1343
// https://github.com/rust-cuda/cuda-sys/blob/3a973786b3482e3fdfd783cd692fbc3c665d5c11/cuda-runtime-sys/src/cuda_runtime.rs#L4638

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
