blocking_type_dict = {
    "__cudaPushCallConfiguration": "NonBlocking",
    "__cudaPopCallConfiguration": "NonBlocking",
    "cublasSetMathMode": "NonBlocking",
    "cublasSetStream_v2": "NonBlocking",
    "cublasSgemm_v2": "GPUBlocking",
    "cublasSgemmStridedBatched": "GPUBlocking",
    "cudaDeviceGetAttribute": "NonBlocking",
    "cudaGetDevice": "NonBlocking",
    "cudaGetLastError": "NonBlocking",
    "cudaLaunchKernel": "GPUBlocking",
    "cudaMemcpyAsync": "GPUBlocking",
    "cudaMemcpyAsyncDeviceToDevice": "GPUBlocking",
    "cudaMemcpyAsyncDeviceToHost": "GPUBlocking",
    "cudaMemcpyAsyncHostToDevice": "GPUBlocking",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": "NonBlocking",
    "cudaPeekAtLastError": "NonBlocking",
    "cudaStreamIsCapturing": "CPUBlocking",
    "cudaStreamSynchronize": "CPUBlocking",
    "cudnnBatchNormalizationBackwardEx": "GPUBlocking",
    "cudnnBatchNormalizationForwardInference": "GPUBlocking",
    "cudnnBatchNormalizationForwardTrainingEx": "GPUBlocking",
    "cudnnConvolutionBackwardData": "GPUBlocking",
    "cudnnConvolutionBackwardFilter": "GPUBlocking",
    "cudnnConvolutionForward": "GPUBlocking",
    "cudnnCreateConvolutionDescriptor": "NonBlocking",
    "cudnnCreateFilterDescriptor": "NonBlocking",
    "cudnnCreateTensorDescriptor": "NonBlocking",
    "cudnnDestroyConvolutionDescriptor": "NonBlocking",
    "cudnnDestroyFilterDescriptor": "NonBlocking",
    "cudnnDestroyTensorDescriptor": "NonBlocking",
    "cudnnGetBatchNormalizationBackwardExWorkspaceSize": "NonBlocking",
    "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize": "NonBlocking",
    "cudnnGetBatchNormalizationTrainingExReserveSpaceSize": "NonBlocking",
    "cudnnSetConvolutionGroupCount": "NonBlocking",
    "cudnnSetConvolutionMathType": "NonBlocking",
    "cudnnSetConvolutionNdDescriptor": "NonBlocking",
    "cudnnSetFilterNdDescriptor": "NonBlocking",
    "cudnnSetTensorNdDescriptor": "NonBlocking",
    "cudnnSetStream": "NonBlocking",
}


def get_blocking_type(api_name: str) -> str:
    # if not find then error
    if api_name not in blocking_type_dict:
        raise ValueError(f"Blocking type '{api_name}' not found in blocking_type_dict.")
    return blocking_type_dict[api_name]

