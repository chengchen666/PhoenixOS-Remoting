#include "timer.h"

/********************************************/
/*                 Utils                    */
/********************************************/

uint64_t rdtscp(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi));
    return ((uint64_t)lo) | (((uint64_t)hi) << 32);
}


/********************************************/
/*                 Timer                    */
/********************************************/

Timer::Timer() {
    elapsed_time = Time();
}

Time Timer::elapsed() {
    return elapsed_time;
}

/********************************************/
/*                CPUTimer                  */
/********************************************/

double CPUTimer::counts_2_us(uint64_t counts) {
    assert(hz != 0);
    return counts * (1000000.0 / hz);
}

CPUTimer::CPUTimer() {
    hz = 0;
    global_start_clock = 0;
    global_start_chrono = std::chrono::steady_clock::now();
    start_clock = 0;
}

void CPUTimer::start_trace() {
    hz = 0;
    global_start_clock = rdtscp();
    global_start_chrono = std::chrono::steady_clock::now();
}

void CPUTimer::end_trace() {
    uint64_t clocks = rdtscp() - global_start_clock;
    double nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - global_start_chrono)
            .count();
    hz = clocks / nanoseconds * 1000000000.0;
    std::cout << "<CPUTimer> Elapsed time: " << nanoseconds / 1000000000.0 << " s" << std::endl;
    std::cout << "<CPUTimer> CPU frequency: " << hz / 1000000000.0 << " GHz" << std::endl;
}

void CPUTimer::start() {
    start_clock = rdtscp();
}

void CPUTimer::stop() {
    elapsed_time = Time(rdtscp() - start_clock);
}


/********************************************/
/*                GPUTimer                  */
/********************************************/

GPUTimer::GPUTimer() {
    cudaEventCreate_entry = reinterpret_cast<cudaEventCreate_ptr>(HOOK_CUDART_SYMBOL("cudaEventCreate"));
    cudaEventDestroy_entry = reinterpret_cast<cudaEventDestroy_ptr>(HOOK_CUDART_SYMBOL("cudaEventDestroy"));
    cudaEventRecord_entry = reinterpret_cast<cudaEventRecord_ptr>(HOOK_CUDART_SYMBOL("cudaEventRecord"));
    cudaEventSynchronize_entry = reinterpret_cast<cudaEventSynchronize_ptr>(HOOK_CUDART_SYMBOL("cudaEventSynchronize"));
    cudaEventElapsedTime_entry = reinterpret_cast<cudaEventElapsedTime_ptr>(HOOK_CUDART_SYMBOL("cudaEventElapsedTime"));

    cudaEventCreate_entry(&start_event);
    cudaEventCreate_entry(&stop_event);
    stream = 0;
}

GPUTimer::~GPUTimer() {
    cudaEventDestroy_entry(start_event);
    cudaEventDestroy_entry(stop_event);
}

void GPUTimer::set_stream(cudaStream_t s) {
    stream = s;
}

cudaStream_t GPUTimer::get_stream() {
    return stream;
}

void GPUTimer::start() {
    cudaEventRecord_entry(start_event, stream);
}

void GPUTimer::stop() {
    cudaEventRecord_entry(stop_event, stream);
    cudaEventSynchronize_entry(stop_event);
    float ms;
    cudaEventElapsedTime_entry(&ms, start_event, stop_event);
    elapsed_time = Time(double(ms) * 1000.0);
}
