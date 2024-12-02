#![expect(non_camel_case_types, non_upper_case_globals)]

macro_rules! success_return_value {
    ($ty:ident::$variant:ident) => {
        const _: () = {
            assert!($ty::$variant as u32 == 0);
            const fn test_num_derive<T: crate::FromPrimitive>() {}
            test_num_derive::<$ty>(); // see `DeriveCallback` in `build.rs`
        };
        impl Default for $ty {
            #[inline(always)]
            fn default() -> Self {
                Self::$variant
            }
        }
    };
}

pub mod cuda {
    include!("bindings/types/cuda.rs");

    const _: () = assert!(CUDA_VERSION >= 11030);

    success_return_value!(CUresult::CUDA_SUCCESS);
}

pub mod cudart {
    include!("bindings/types/cudart.rs");

    const _: () = assert!(CUDART_VERSION >= 11030);

    success_return_value!(cudaError_t::cudaSuccess);
}

pub mod nvml {
    include!("bindings/types/nvml.rs");

    success_return_value!(nvmlReturn_t::NVML_SUCCESS);
}

pub mod cudnn {
    include!("bindings/types/cudnn.rs");

    success_return_value!(cudnnStatus_t::CUDNN_STATUS_SUCCESS);
}

pub mod cublas {
    include!("bindings/types/cublas.rs");

    success_return_value!(cublasStatus_t::CUBLAS_STATUS_SUCCESS);
}

pub mod cublasLt {
    include!("bindings/types/cublasLt.rs");

    success_return_value!(cublasStatus_t::CUBLAS_STATUS_SUCCESS);
}

pub mod nvrtc {
    include!("bindings/types/nvrtc.rs");

    success_return_value!(nvrtcResult::NVRTC_SUCCESS);
}

pub mod nccl {
    include!("bindings/types/nccl.rs");

    const _: () = assert!(NCCL_VERSION_CODE >= 21602, "run `apt install libnccl2 libnccl-dev`");

    success_return_value!(ncclResult_t::ncclSuccess);
}
