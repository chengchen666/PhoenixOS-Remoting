remoting_type_dict = {
    # [remoting_type, payload_backward]
    "__cudaPushCallConfiguration": ["LOCAL", 0],
    "__cudaPopCallConfiguration": ["LOCAL", 0],
    "cublasSetMathMode": ["ASYNC", 0],
    "cublasSetStream_v2": ["ASYNC", 0],
    "cublasSgemm_v2": ["ASYNC", 0],
    "cublasSgemmStridedBatched": ["ASYNC", 0],
    "cudaDeviceGetAttribute": "SYNC",
    "cudaGetDevice": ["LOCAL", 0],
    "cudaGetLastError": ["LOCAL", 0],
    "cudaLaunchKernel": ["ASYNC", 0],
    "cudaMemcpyAsync": ["ASYNC", 0],
    "cudaMemcpyAsyncDeviceToDevice": ["ASYNC", 0],
    "cudaMemcpyAsyncDeviceToHost": ["ASYNC", 0],
    "cudaMemcpyAsyncHostToDevice": ["ASYNC", 0],
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": ["LOCAL", 0],
    "cudaPeekAtLastError": ["LOCAL", 0],
    "cudaStreamIsCapturing": ["SYNC", 8],
    "cudaStreamSynchronize": ["SYNC", 4],
    "cudnnBatchNormalizationBackwardEx": ["SYNC", 4],
    "cudnnBatchNormalizationForwardInference": ["ASYNC", 0],
    "cudnnBatchNormalizationForwardTrainingEx": ["SYNC", 4],
    "cudnnConvolutionBackwardData": ["SYNC", 4],
    "cudnnConvolutionBackwardFilter": ["SYNC", 4],
    "cudnnConvolutionForward": ["ASYNC", 0],
    "cudnnCreateConvolutionDescriptor": ["ASYNC", 0],
    "cudnnCreateFilterDescriptor": ["ASYNC", 0],
    "cudnnCreateTensorDescriptor": ["ASYNC", 0],
    "cudnnDestroyConvolutionDescriptor": ["ASYNC", 0],
    "cudnnDestroyFilterDescriptor": ["ASYNC", 0],
    "cudnnDestroyTensorDescriptor": ["ASYNC", 0],
    "cudnnGetBatchNormalizationBackwardExWorkspaceSize": ["SYNC", 12],
    "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize": ["SYNC", 12],
    "cudnnGetBatchNormalizationTrainingExReserveSpaceSize": ["SYNC", 12],
    "cudnnSetConvolutionGroupCount": ["ASYNC", 0],
    "cudnnSetConvolutionMathType": ["ASYNC", 0],
    "cudnnSetConvolutionNdDescriptor": ["ASYNC", 0],
    "cudnnSetFilterNdDescriptor": ["ASYNC", 0],
    "cudnnSetTensorNdDescriptor": ["ASYNC", 0],
    "cudnnSetStream": ["ASYNC", 0],
}


def get_remoting_type(api_name: str) -> str:
    # if not find then error
    if api_name not in remoting_type_dict:
        raise ValueError(f"Execution type '{api_name}' not found in remoting_type_dict.")
    return remoting_type_dict[api_name]

