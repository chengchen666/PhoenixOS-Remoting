use cudasys::cuda::*;
use std::os::raw::*;

pub fn cu_launch_kernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    args: &[u8],
) -> CUresult {
    let args_len = args.len();
    let extra_array: [*mut c_void; 5] = [
        1 as _, // CU_LAUNCH_PARAM_BUFFER_POINTER
        args.as_ptr() as _,
        2 as _, // CU_LAUNCH_PARAM_BUFFER_SIZE
        &raw const args_len as _,
        std::ptr::null_mut(), // CU_LAUNCH_PARAM_END
    ];
    unsafe {
        cuLaunchKernel(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            std::ptr::null_mut(),
            extra_array.as_ptr().cast_mut(),
        )
    }
}
