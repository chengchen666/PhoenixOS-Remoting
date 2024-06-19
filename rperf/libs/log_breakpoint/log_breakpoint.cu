#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    }

// This function is used to insert a breakpoint in the pytorch cuda api calls.
// `cudaDeviceSynchronize` is never used in the pytorch codebase, so it is safe
// to use it as a breakpoint.
extern "C" void log_breakpoint() {
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Log breakpoint reached, subsequent logs are traced." << std::endl;
}

int main() {
    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(0));

    // Log before the breakpoint
    std::cout << "Log before the breakpoint." << std::endl;

    // Insert the log breakpoint
    log_breakpoint();

    // Log after the breakpoint
    std::cout << "Log after the breakpoint." << std::endl;

    return 0;
}
