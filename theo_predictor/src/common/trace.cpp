#include "trace.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>

int on_trace = 0;

static uint64_t hz = 0;
static uint64_t global_start_clock = 0;
static std::chrono::time_point<std::chrono::steady_clock> global_start_chrono;
static double nanoseconds;

extern "C" void startTrace() {
    std::cout << "start Tracing..." << std::endl;
    on_trace = 1;
    global_start_clock = rdtscp();
    global_start_chrono = std::chrono::steady_clock::now();
}

extern "C" void endTrace() {
    std::cout << "end Tracing..." << std::endl;
    on_trace = 0;
    uint64_t clocks = rdtscp() - global_start_clock;
    nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - global_start_chrono)
            .count();
    hz = clocks / (nanoseconds / 1000000000.0);
    std::cout << "CPU time: " << nanoseconds / 1000000000.0 << " s" << std::endl;
    std::cout << "CPU frequency: " << hz / 1000000000.0 << " GHz" << std::endl;
}

uint64_t rdtscp(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi));
    return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}

uint64_t cycles_2_ns(uint64_t cycles) {
    assert(hz != 0);
    return cycles * (1000000000.0 / hz);
}

std::map<std::string, int> *api_dict;
std::vector<APIRecord> *api_records;

static void add_api(int api_id, const std::string &api_name) {
    (*api_dict)[api_name] = api_id;
}

TraceProfile::TraceProfile(const std::string &name) {
    if (!on_trace) {
        return;
    }
    api_name = name;
    call_start = rdtscp();
}

TraceProfile::~TraceProfile() {
    if (!on_trace) {
        return;
    }
    api_records->emplace_back(api_name, (rdtscp() - call_start));
}

void __attribute__((constructor)) trace_init(void) {
    api_dict = new std::map<std::string, int>();
    int id = 0;
    add_api(++id, "__cudaPushCallConfiguration");
    add_api(++id, "__cudaPopCallConfiguration");
    add_api(++id, "__cudaRegisterFatBinary");
    add_api(++id, "__cudaRegisterFatBinaryEnd");
    add_api(++id, "__cudaRegisterFunction");
    add_api(++id, "__cudaRegisterVar");
    add_api(++id, "__cudaUnregisterFatBinary");
    add_api(++id, "cuDevicePrimaryCtxGetState");
    add_api(++id, "cublasCreate_v2");
    add_api(++id, "cublasGemmEx");
    add_api(++id, "cublasGemmStridedBatchedEx");
    add_api(++id, "cublasGetMathMode");
    add_api(++id, "cublasSetMathMode");
    add_api(++id, "cublasSetStream_v2");
    add_api(++id, "cublasSgemmStridedBatched");
    add_api(++id, "cublasSgemm_v2");
    add_api(++id, "cudaDeviceGetAttribute");
    add_api(++id, "cudaFuncGetAttributes");
    add_api(++id, "cudaFree");
    add_api(++id, "cudaGetDevice");
    add_api(++id, "cudaGetDeviceCount");
    add_api(++id, "cudaGetDeviceProperties");
    add_api(++id, "cudaGetLastError");
    add_api(++id, "cudaHostAlloc");
    add_api(++id, "cudaLaunchKernel");
    add_api(++id, "cudaMalloc");
    add_api(++id, "cudaMemcpyAsyncHostToDevice");
    add_api(++id, "cudaMemcpyAsyncDeviceToHost");
    add_api(++id, "cudaMemcpyAsyncDeviceToDevice");
    add_api(++id, "cudaMemsetAsync");
    add_api(++id, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
    add_api(++id, "cudaPeekAtLastError");
    add_api(++id, "cudaPointerGetAttributes");
    add_api(++id, "cudaSetDevice");
    add_api(++id, "cudaStreamIsCapturing");
    add_api(++id, "cudaStreamSynchronize");
    add_api(++id, "cudnnBatchNormalizationBackwardEx");
    add_api(++id, "cudnnBatchNormalizationForwardInference");
    add_api(++id, "cudnnBatchNormalizationForwardTrainingEx");
    add_api(++id, "cudnnConvolutionBackwardData");
    add_api(++id, "cudnnConvolutionBackwardFilter");
    add_api(++id, "cudnnConvolutionForward");
    add_api(++id, "cudnnCreate");
    add_api(++id, "cudnnCreateConvolutionDescriptor");
    add_api(++id, "cudnnCreateFilterDescriptor");
    add_api(++id, "cudnnCreateTensorDescriptor");
    add_api(++id, "cudnnDestroyConvolutionDescriptor");
    add_api(++id, "cudnnDestroyFilterDescriptor");
    add_api(++id, "cudnnDestroyTensorDescriptor");
    add_api(++id, "cudnnGetBatchNormalizationBackwardExWorkspaceSize");
    add_api(++id, "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
    add_api(++id, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
    add_api(++id, "cudnnGetConvolutionBackwardDataAlgorithm_v7");
    add_api(++id, "cudnnGetConvolutionBackwardFilterAlgorithm_v7");
    add_api(++id, "cudnnGetConvolutionForwardAlgorithm_v7");
    add_api(++id, "cudnnInitTransformDest");
    add_api(++id, "cudnnSetConvolutionGroupCount");
    add_api(++id, "cudnnSetConvolutionMathType");
    add_api(++id, "cudnnSetConvolutionNdDescriptor");
    add_api(++id, "cudnnSetFilterNdDescriptor");
    add_api(++id, "cudnnSetStream");
    add_api(++id, "cudnnSetTensorNdDescriptor");
    add_api(++id, "cudnnSetTensorNdDescriptorEx");
    add_api(++id, "cudnnSetTensorTransformDescriptor");
    add_api(++id, "nvmlDeviceGetCount_v2");
    add_api(++id, "nvmlInitWithFlags");
    add_api(++id, "nvmlInit_v2");

    api_records = new std::vector<APIRecord>();
}

static void print_api_records() {
    std::map<std::string, int> api_count;
    std::map<std::string, uint64_t> api_time;
    for (auto &api_record : *api_records) {
        api_record.interval = cycles_2_ns(api_record.interval);
        if (api_dict->find(api_record.api_name) == api_dict->end()) {
            std::cout << "[ERROR] API " << api_record.api_name << " not set in dict!" << std::endl;
            exit(1);
        }
        std::string &api_name = api_record.api_name;
        if (api_count.find(api_name) == api_count.end()) {
            api_count[api_name] = 0;
            api_time[api_name] = 0;
        }
        api_count[api_name]++;
        api_time[api_name] += api_record.interval;
        if (api_record.api_name.find("__cuda") != std::string::npos ||
            api_record.api_name.find("rpc_") != std::string::npos) {
            api_record.api_name = "";
        }
    }

    std::string rperf_log_path = "vanilla_rperf.log";
    std::ofstream out(rperf_log_path, std::ios::out);
    printf("RPerf log path: %s\n", rperf_log_path.c_str());
    out << "Elapsed time(s):"<< std::endl << nanoseconds / 1000000000.0 << std::endl;
    out << "API: name, driver(ns)" << std::endl;
    for (auto &api : api_time) {
        out << api.first << " " << api.second << std::endl;
    }

    // std::cout << std::endl << std::endl << "API Lists:" << std::endl;
    // for (auto &api : api_count) {
    //     std::cout << api.first << " " << api.second << std::endl;
    // }

    // std::cout << std::endl << std::endl << "API Traces: api, interval(ns)" << std::endl;
    // for (auto &api_record : *api_records) {
    //     if (api_record.api_name == "") {
    //         continue;
    //     }
    //     std::cout << api_record.api_name << " " << api_record.interval << std::endl;
    // }
}

void __attribute__((destructor)) trace_deinit(void) {
    print_api_records();
    delete api_dict;
    delete api_records;
}
