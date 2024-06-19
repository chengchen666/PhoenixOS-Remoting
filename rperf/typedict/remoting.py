remoting_type_dict = {
    "__cudaPushCallConfiguration": "LOCAL",
    "__cudaPopCallConfiguration": "LOCAL",
    "cublasSetMathMode": "ASYNC",
    "cublasSetStream_v2": "ASYNC",
    "cublasSgemm_v2": "ASYNC",
    "cublasSgemmStridedBatched": "ASYNC",
    "cudaDeviceGetAttribute": "SYNC",
    "cudaGetDevice": "LOCAL",
    "cudaGetLastError": "LOCAL",
    "cudaLaunchKernel": "ASYNC",
    "cudaMemcpyAsyncDeviceToDevice": "ASYNC",
    "cudaMemcpyAsyncDeviceToHost": "ASYNC",
    "cudaMemcpyAsyncHostToDevice": "ASYNC",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags": "LOCAL",
    "cudaPeekAtLastError": "LOCAL",
    "cudaStreamIsCapturing": "SYNC",
    "cudaStreamSynchronize": "SYNC",
    "cudnnBatchNormalizationForwardInference": "ASYNC",
    "cudnnConvolutionForward": "ASYNC",
    "cudnnCreateConvolutionDescriptor": "ASYNC",
    "cudnnCreateFilterDescriptor": "ASYNC",
    "cudnnCreateTensorDescriptor": "ASYNC",
    "cudnnDestroyConvolutionDescriptor": "ASYNC",
    "cudnnDestroyFilterDescriptor": "ASYNC",
    "cudnnDestroyTensorDescriptor": "ASYNC",
    "cudnnSetConvolutionGroupCount": "ASYNC",
    "cudnnSetConvolutionMathType": "ASYNC",
    "cudnnSetConvolutionNdDescriptor": "ASYNC",
    "cudnnSetFilterNdDescriptor": "ASYNC",
    "cudnnSetTensorNdDescriptor": "ASYNC",
    "cudnnSetStream": "ASYNC",
}


def get_remoting_type(api_name: str) -> str:
    # if not find then error
    if api_name not in remoting_type_dict:
        raise ValueError(f"Execution type '{api_name}' not found in remoting_type_dict.")
    return remoting_type_dict[api_name]

