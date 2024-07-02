#ifndef TRACE_H
#define TRACE_H

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "macro_common.h"
#include "timer.h"

#define HOOK_TRACE_SWITCH

extern int on_trace;
extern int on_separate_kernel_wrapper;
extern CPUTimer cpu_timer;
extern GPUTimer gpu_timer;
extern uint64_t last_return_ts;
extern "C" {
void startTrace();
void endTrace();
}

class APIRecord {
public:
    std::string api_name;
    Time interval;

    APIRecord(const std::string &name, const Time itv) : api_name(name), interval(itv) {}
};

extern std::map<std::string, int> *api_dict;
extern std::vector<APIRecord> *api_records;

class TraceProfile {
public:
    TraceProfile(const std::string &name);
    ~TraceProfile();

private:
    std::string api_name;
    TraceProfile(const TraceProfile &) = delete;
    void operator=(const TraceProfile &) = delete;
};

void push_breakpoint();

#ifdef HOOK_TRACE_SWITCH
#define HOOK_TRACE_PROFILE(name) TraceProfile _tp_##name_(name)
#else
#define HOOK_TRACE_PROFILE(name)
#endif

#endif  // TRACE_H
