#ifndef TRACE_H
#define TRACE_H

#include <map>
#include <string>
#include <vector>
#include <cstdint>

#define HOOK_TRACE_SWITCH

extern int on_trace;
extern "C" {
void startTrace();
void endTrace();
}

uint64_t rdtscp(void);

uint64_t cycles_2_ns(uint64_t cycles);

class APIRecord {
public:
    std::string api_name;
    uint64_t interval;

    APIRecord(const std::string &name, const uint64_t itv) : api_name(name), interval(itv) {}
};

extern std::map<std::string, int> *api_dict;
extern std::vector<APIRecord> *api_records;

class TraceProfile {
public:
    TraceProfile(const std::string &name);
    ~TraceProfile();

private:
    std::string api_name;
    uint64_t call_start;

    TraceProfile(const TraceProfile &) = delete;
    void operator=(const TraceProfile &) = delete;
};

#ifdef HOOK_TRACE_SWITCH
#define HOOK_TRACE_PROFILE(name) TraceProfile _tp_##name_(name)
#else
#define HOOK_TRACE_PROFILE(name)
#endif

#endif  // TRACE_H
