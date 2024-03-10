#include <cuda_runtime.h>
#include <iostream>
#include <nvml.h>

int main()
{
    // const int iterations = 1;
    nvmlReturn_t result;
    result = nvmlInit_v2();
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    // result = nvmlInitWithFlags(NVML_INIT_FLAG_NO_ATTACH);
    // if (NVML_SUCCESS != result)
    // {
    //     std::cout << "Failed to initialize NVML with NO_ATTACH: " << nvmlErrorString(result) << std::endl;
    //     return 1;
    // }

    unsigned int device_count;
    result = nvmlDeviceGetCount_v2(&device_count);
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " devices" << std::endl;

    return 0;
}
