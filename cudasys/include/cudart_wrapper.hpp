#include <cmath>
#include <cstring>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDACC_RTC__
#define __CUDACC__
#include "crt/device_functions.h"  // __cudaPushCallConfiguration
#undef __CUDACC__
#undef __CUDACC_RTC__

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
#include "crt/host_runtime.h"  // the rest __cuda* functions
