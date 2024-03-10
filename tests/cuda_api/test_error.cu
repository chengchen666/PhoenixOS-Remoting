#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

int main()
{
    // const int iterations = 1;
    cudaError_t error;

    error = cudaGetLastError();
    std::cout << "Error: " << cudaGetErrorString(error) << std::endl;

    error = cudaPeekAtLastError();
    std::cout << "Error: " << cudaGetErrorString(error) << std::endl;

    return 0;
}
