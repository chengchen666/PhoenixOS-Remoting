use cudasys::cuda::*;
use std::os::raw::*;

pub fn cu_launch_kernel(
    #[cfg(feature = "phos")] pos_cuda_ws: *mut c_void,
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
    #[cfg(not(feature = "phos"))]
    unsafe {
        let args_len = args.len();
        let extra_array: [*mut c_void; 5] = [
            1 as _, // CU_LAUNCH_PARAM_BUFFER_POINTER
            args.as_ptr() as _,
            2 as _, // CU_LAUNCH_PARAM_BUFFER_SIZE
            &raw const args_len as _,
            std::ptr::null_mut(), // CU_LAUNCH_PARAM_END
        ];
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
    #[cfg(feature = "phos")]
    {
        use super::*;
        use cudasys::cudart::dim3;

        let gridDim = dim3 { x: gridDimX, y: gridDimY, z: gridDimZ };
        let blockDim = dim3 { x: blockDimX, y: blockDimY, z: blockDimZ };
        let sharedMem = sharedMemBytes as usize;
        CUresult::from_i32(call_pos_process(
            pos_cuda_ws,
            239, // cudaLaunchKernel
            0,
            &[
                &raw const f as usize,
                size_of_val(&f),
                &raw const gridDim as usize,
                size_of_val(&gridDim),
                &raw const blockDim as usize,
                size_of_val(&blockDim),
                args.as_ptr() as usize,
                args.len(),
                &raw const sharedMem as usize,
                size_of_val(&sharedMem),
                &raw const hStream as usize,
                size_of_val(&hStream),
            ],
        ))
        .expect("Illegal result ID")
    }
}
