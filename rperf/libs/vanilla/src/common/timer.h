#ifndef TIMER_H
#define TIMER_H

#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>

#include "hook.h"
#include "../cudart/cudart_subset.h"

uint64_t rdtscp(void);

class Time {
public:
    enum class Type { COUNT, TIME } type;
    union Value {
        double time;
        uint64_t count;
    } value;
    Time() : type(Type::TIME) {
        value.time = 0.0;
    }
    Time(double t) : type(Type::TIME) {
        value.time = t;
    }
    Time(uint64_t c) : type(Type::COUNT) {
        value.count = c;
    }
    Time(const Time &t) : type(t.type) {
        value = t.value;
    }
    // print
    friend std::ostream &operator<<(std::ostream &os, const Time &t) {
        assert(t.type == Time::Type::TIME);
        os << t.value.time;
        return os;
    }
};

class Timer {
public:
    Timer();
    ~Timer() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    Time elapsed();

protected:
    Time elapsed_time;
};

class CPUTimer : public Timer {
public:
    CPUTimer();
    ~CPUTimer() = default;
    void start();
    void stop();
    void start_trace();
    void end_trace();
    double counts_2_us(uint64_t counts);

private:
    uint64_t hz;
    uint64_t global_start_clock;
    std::chrono::time_point<std::chrono::steady_clock> global_start_chrono;
    uint64_t start_clock;
};

class GPUTimer : public Timer {
public:
    GPUTimer();
    ~GPUTimer();
    void start();
    void stop();
    void set_stream(cudaStream_t s);
    cudaStream_t get_stream();

private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cudaStream_t stream;

    using cudaEventCreate_ptr = cudaError_t (*)(cudaEvent_t *);
    using cudaEventDestroy_ptr = cudaError_t (*)(cudaEvent_t);
    using cudaEventRecord_ptr = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    using cudaEventSynchronize_ptr = cudaError_t (*)(cudaEvent_t);
    using cudaEventElapsedTime_ptr = cudaError_t (*)(float *, cudaEvent_t, cudaEvent_t);
    cudaEventCreate_ptr cudaEventCreate_entry;
    cudaEventDestroy_ptr cudaEventDestroy_entry;
    cudaEventRecord_ptr cudaEventRecord_entry;
    cudaEventSynchronize_ptr cudaEventSynchronize_entry;
    cudaEventElapsedTime_ptr cudaEventElapsedTime_entry;
};

#endif  // TIMER_H