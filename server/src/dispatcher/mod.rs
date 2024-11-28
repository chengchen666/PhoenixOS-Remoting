#![expect(non_snake_case)]

use codegen::cuda_hook_exe;
use log::error;
use network::type_impl::{recv_slice, send_slice, MemPtr};
use network::{CommChannel, Transportable};

use crate::ServerWorker;

mod cuda_exe;
mod cuda_exe_custom;
mod cudart_exe;
mod cudart_exe_custom;
mod nvml_exe;
mod cudnn_exe_custom;
mod cudnn_exe;
mod cublas_exe;

use cuda_exe::*;
use cuda_exe_custom::*;
use cudart_exe::*;
use cudart_exe_custom::*;
use nvml_exe::*;
use cudnn_exe_custom::*;
use cudnn_exe::*;
use cublas_exe::*;

pub fn dispatch<C: CommChannel>(proc_id: i32, server: &mut ServerWorker<C>) {
    // let start = network::NsTimestamp::now();
    let func: fn(&mut ServerWorker<C>) = match proc_id {
        0 => cudaGetDeviceExe,
        1 => cudaSetDeviceExe,
        2 => cudaGetDeviceCountExe,
        3 => cudaGetLastErrorExe,
        4 => cudaPeekAtLastErrorExe,
        5 => cudaStreamSynchronizeExe,
        6 => cudaMallocExe,
        7 => cudaMemcpyExe,
        8 => cudaFreeExe,
        9 => cudaStreamIsCapturingExe,
        #[cfg(not(cuda_version = "12"))]
        10 => cudaGetDevicePropertiesExe,
        11 => unimplemented!("cudaMallocManaged"),
        12 => cudaPointerGetAttributesExe,
        13 => unimplemented!("cudaHostAlloc"),
        14 => cudaFuncGetAttributesExe,
        15 => cudaDeviceGetStreamPriorityRangeExe,
        16 => cudaMemsetAsyncExe,
        17 => cudaGetErrorStringExe,
        #[cfg(cuda_version = "12")]
        18 => cudaGetDeviceProperties_v2Exe,
        100 => __cudaRegisterFatBinaryExe,
        101 => unimplemented!("__cudaUnregisterFatBinary"),
        102 => __cudaRegisterFunctionExe,
        103 => __cudaRegisterVarExe,
        200 => cudaLaunchKernelExe,
        300 => cuDevicePrimaryCtxGetStateExe,
        500 => unimplemented!("cuGetProcAddress"),
        501 => cuDriverGetVersionExe,
        502 => cuInitExe,
        503 => unimplemented!("cuGetExportTable"),
        504 => cuCtxGetCurrentExe,
        1000 => nvmlInit_v2Exe,
        1001 => nvmlDeviceGetCount_v2Exe,
        1002 => nvmlInitWithFlagsExe,
        1500 => cudnnCreateExe,
        1501 => cudnnCreateTensorDescriptorExe,
        1502 => cudnnSetTensor4dDescriptorExe,
        1503 => cudnnCreateActivationDescriptorExe,
        1504 => cudnnSetActivationDescriptorExe,
        1505 => unimplemented!("cudnnActivationForwardExe"),
        1506 => cudnnDestroyExe,
        1507 => cudnnSetConvolution2dDescriptorExe,
        1508 => cudnnSetStreamExe,
        1509 => cudnnSetTensorNdDescriptorExe,
        1510 => cudnnDestroyTensorDescriptorExe,
        1511 => cudnnCreateFilterDescriptorExe,
        1512 => cudnnDestroyFilterDescriptorExe,
        1513 => cudnnSetFilterNdDescriptorExe,
        1514 => cudnnCreateConvolutionDescriptorExe,
        1515 => cudnnDestroyConvolutionDescriptorExe,
        1516 => cudnnSetConvolutionNdDescriptorExe,
        1517 => cudnnSetConvolutionGroupCountExe,
        1518 => cudnnSetConvolutionMathTypeExe,
        1519 => cudnnSetConvolutionReorderTypeExe,
        1520 => cudnnGetConvolutionForwardAlgorithm_v7Exe,
        1521 => cudnnConvolutionForwardExe,
        1522 => cudnnGetBatchNormalizationForwardTrainingExWorkspaceSizeExe,
        1523 => cudnnGetBatchNormalizationTrainingExReserveSpaceSizeExe,
        1524 => cudnnBatchNormalizationForwardTrainingExExe,
        1525 => cudnnGetBatchNormalizationBackwardExWorkspaceSizeExe,
        1526 => cudnnBatchNormalizationBackwardExExe,
        1527 => cudnnGetConvolutionBackwardDataAlgorithm_v7Exe,
        1528 => cudnnConvolutionBackwardDataExe,
        1529 => cudnnGetConvolutionBackwardFilterAlgorithm_v7Exe,
        1530 => cudnnConvolutionBackwardFilterExe,
        1531 => cudnnBatchNormalizationForwardInferenceExe,
        1532 => cudnnSetFilter4dDescriptorExe,
        1533 => cudnnGetConvolutionNdForwardOutputDimExe,
        1534 => cudnnGetConvolutionForwardWorkspaceSizeExe,
        1535 => cudnnGetErrorStringExe,
        2000 => cublasCreate_v2Exe,
        2001 => cublasDestroy_v2Exe,
        2002 => cublasSetStream_v2Exe,
        2003 => cublasSetMathModeExe,
        2004 => cublasSgemm_v2Exe,
        2005 => cublasSgemmStridedBatchedExe,
        2006 => cublasGetMathModeExe,
        2007 => cublasGemmExExe,
        2008 => cublasGemmStridedBatchedExExe,
        2009 => cublasSetWorkspace_v2Exe,
        other => {
            error!(
                "[{}:{}] invalid proc_id: {}",
                std::file!(),
                std::line!(),
                other
            );
            return;
        }
    };
    func(server);
    // let end = network::NsTimestamp::now();
    // let elapsed = (end.sec_timestamp - start.sec_timestamp) as f64 * 1000000000.0
    //             + (end.ns_timestamp as i32 - start.ns_timestamp as i32) as f64;
    // info!("exe: {}", elapsed / 1000.0);
}
