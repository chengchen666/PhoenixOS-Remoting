#ifndef _ELF_INTERFACES_H_
#define _ELF_INTERFACES_H_

#include "uthash.h"

struct __attribute__((__packed__)) fat_header {
    uint32_t magic;
    uint32_t version;
    uint64_t text;      // points to first text section
    uint64_t data;      // points to outside of the file
    uint64_t unknown;
    uint64_t text2;     // points to second text section
    uint64_t zero;
};

int elf2_init(void);
int elf2_get_fatbin_info(const struct fat_header *fatbin, uint8_t** fatbin_mem, size_t* fatbin_size);
int elf2_parameter_info(void* memory, size_t memsize);
void* elf2_symbol_address(const char *symbol);

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

kernel_info_t* find_kernel_host_func(const void* func);
kernel_info_t* find_kernel_name(const char* name);
void add_kernel_host_func(const void* func, kernel_info_t* info);
void add_kernel_name(const char* name, kernel_info_t* info);

void kernel_infos_free();
int utils_parameter_info(char *path);
kernel_info_t* utils_search_info(const char *kernelname);

#endif //_ELF_INTERFACES_H_
