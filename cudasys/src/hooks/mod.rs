#![allow(dead_code)] // doesn't work well with expect
#![expect(unused_variables)]
#![expect(clippy::too_many_arguments)]

mod cuda;
mod cudart;
mod nvml;
mod cudnn;
mod cublas;
mod cublasLt;
mod nvrtc;
mod nccl;
