#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

int main()
{
    // get device
    int device = 0;
    cudaGetDevice(&device);
    // call cuDevicePrimaryCtxGetState
    CUdevice dev = 0;
    unsigned int flags;
    int active;
    cuDevicePrimaryCtxGetState(dev, &flags, &active);
    std::cout << "flags: " << flags << ", active: " << active << std::endl;

    return 0;
}
