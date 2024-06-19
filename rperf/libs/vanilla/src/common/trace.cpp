#include "trace.h"

int on_trace = 0;
CPUTimer cpu_timer;
GPUTimer gpu_timer;
uint64_t last_return_ts;
std::string rperf_log_path;

extern "C" void startTrace() {
    std::cout << "start Tracing..." << std::endl;
    on_trace = 1;
    rperf_log_path = "vanilla_rperf.log";
    
    const char* env_var = std::getenv("SEPARATE_KERNEL_WRAPPER");
    if (env_var && (std::string(env_var) == "1" ||
        std::string(env_var) == "on" || std::string(env_var) == "ON")) {
        std::cout << "start separating kernel wrapper..." << std::endl;
        on_separate_kernel_wrapper = 1;
        rperf_log_path = "";
    }

    cpu_timer.start_trace();
    last_return_ts = rdtscp();
}

extern "C" void endTrace() {
    if (on_separate_kernel_wrapper) {
        std::cout << "end separating kernel wrapper..." << std::endl;
        on_separate_kernel_wrapper = 0;
    }

    std::cout << "end Tracing..." << std::endl;
    cpu_timer.end_trace();
    on_trace = 0;
}

std::map<std::string, int> *api_dict;
std::vector<APIRecord> *api_records;
std::vector<Time> *gaps;

static void add_api(int api_id, const std::string &api_name) {
    (*api_dict)[api_name] = api_id;
}

TraceProfile::TraceProfile(const std::string &name) {
    if (!on_trace) {
        return;
    }
    api_name = name;
    gaps->emplace_back(Time(rdtscp() - last_return_ts));
    // if (HOOK_LIKELY(api_name != "cudaLaunchKernel" && api_name != "cudaStreamSynchronize"
    //                     && api_name != "cublasSgemm_v2" && api_name != "cublasSgemmStridedBatched")) {
    //     cpu_timer.start();
    // } else {
    //     gpu_timer.start();
    // }
    cpu_timer.start();
}

TraceProfile::~TraceProfile() {
    if (!on_trace) {
        return;
    }
    // if (HOOK_LIKELY(api_name != "cudaLaunchKernel" && api_name != "cudaStreamSynchronize"
    //                     && api_name != "cublasSgemm_v2" && api_name != "cublasSgemmStridedBatched")) {
    //     cpu_timer.stop();
    //     api_records->emplace_back(api_name, cpu_timer.elapsed());
    // } else {
    //     gpu_timer.stop();
    //     // std::cout << "GPU Timer: " << gpu_timer.elapsed() << std::endl;
    //     api_records->emplace_back(api_name, gpu_timer.elapsed());
    // }
    cpu_timer.stop();
    api_records->emplace_back(api_name, cpu_timer.elapsed());
    last_return_ts = rdtscp();
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
    gaps = new std::vector<Time>();
}

static void print_traces() {
    // std::map<std::string, int> api_count;
    // std::map<std::string, uint64_t> api_time;
    // for (auto &api_record : *api_records) {
    //     api_record.interval = cycles_2_ns(api_record.interval);
    //     if (api_dict->find(api_record.api_name) == api_dict->end()) {
    //         std::cout << "[ERROR] API " << api_record.api_name << " not set in dict!" << std::endl;
    //         exit(1);
    //     }
    //     std::string &api_name = api_record.api_name;
    //     if (api_count.find(api_name) == api_count.end()) {
    //         api_count[api_name] = 0;
    //         api_time[api_name] = 0;
    //     }
    //     api_count[api_name]++;
    //     api_time[api_name] += api_record.interval;
    //     if (api_record.api_name.find("__cuda") != std::string::npos ||
    //         api_record.api_name.find("rpc_") != std::string::npos) {
    //         api_record.api_name = "";
    //     }
    // }

    for (auto &api_record : *api_records) {
        if (api_record.interval.type == Time::Type::COUNT) {
            api_record.interval = Time(cpu_timer.counts_2_us(api_record.interval.value.count));
        }
    }

    for (auto &gap : *gaps) {
        if (gap.type == Time::Type::COUNT) {
            gap = Time(cpu_timer.counts_2_us(gap.value.count));
        }
    }

    if (rperf_log_path.empty()) {
        return;
    }
    std::ofstream out(rperf_log_path, std::ios::out);
    printf("RPerf log path: %s\n", rperf_log_path.c_str());

    double total_exe_time = 0, total_gap_time = 0;

    for (size_t i = 0; i < api_records->size(); i++) {
        out << api_records->at(i).api_name << ", " << api_records->at(i).interval << ", " << gaps->at(i) << std::endl;
        total_exe_time += api_records->at(i).interval.value.time;
        total_gap_time += gaps->at(i).value.time;
    }
    std::cout << "Total execution time: " << total_exe_time << " us" << std::endl;
    std::cout << "Total gap time: " << total_gap_time << " us" << std::endl;
    std::cout << "Total time: " << total_exe_time + total_gap_time << " us" << std::endl;
    out.close();
}

void __attribute__((destructor)) trace_deinit(void) {
    assert(api_records->size() == gaps->size());
    print_traces();
    delete api_dict;
    delete api_records;
    delete gaps;
}

#include "hook.h"
#include "../cudart/cudart_subset.h"

int on_separate_kernel_wrapper = 0;
void push_breakpoint() {
    if (!on_trace || !on_separate_kernel_wrapper) {
        return;
    }
    // This function is used to insert a breakpoint in the pytorch cuda api calls.
    // `cudaDeviceSynchronize` is never used in the pytorch codebase, so it is safe
    // to use it as a breakpoint.
    using func_ptr = cudaError_t (*)();
    static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceSynchronize"));
    HOOK_CHECK(func_entry);
    func_entry();
}
