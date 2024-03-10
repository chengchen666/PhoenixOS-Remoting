#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

int main()
{
    const int iterations = 1;
    int count = 0;
    cudaError_t error;

    error = cudaStreamSynchronize(0);
    std::cout << "cudaStreamSynchronize(0) returned " << error << std::endl;

    cudaStreamCaptureStatus status;
    cudaStreamIsCapturing(0, &status);
    std::cout << "cudaStreamIsCapturing(0) returned " << status << std::endl;

    return 0;
}
