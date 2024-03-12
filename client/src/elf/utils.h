#ifndef _CPU_UTILS_H_
#define _CPU_UTILS_H_

#include <stdint.h>
#include "uthash.h"
#include "list.h"
#include "log.h"

#ifndef LOG_LEVEL
    #define LOG_LEVEL LOG_ERROR
#endif //LOG_LEVEL

typedef struct kernel_info {
    char *name;
    size_t param_size;
    size_t param_num;
    uint16_t *param_offsets;
    uint16_t *param_sizes;
    void *host_fun;
    UT_hash_handle hh_host_func;        /* handle for host_func hash table */
    UT_hash_handle hh_name;            /* handle for name hash table */
} kernel_info_t;

extern kernel_info_t* func_ptr_to_kernel_infos;
extern kernel_info_t* name_to_kernel_infos;

kernel_info_t* find_kernel_host_func(const void* func);
kernel_info_t* find_kernel_name(const char* name);
void add_kernel_host_func(const void* func, kernel_info_t* info);
void add_kernel_name(const char* name, kernel_info_t* info);

void kernel_infos_free();
int cpu_utils_parameter_info(char *path);
kernel_info_t* utils_search_info(const char *kernelname);

#endif //_CPU_UTILS_H_
