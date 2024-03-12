#ifndef _CPU_UTILS_H_
#define _CPU_UTILS_H_

#include <stdint.h>
#include "list.h"
#include "log.h"
#include "interfaces.h"

#ifndef LOG_LEVEL
    #define LOG_LEVEL LOG_ERROR
#endif //LOG_LEVEL

extern kernel_info_t* func_ptr_to_kernel_infos;
extern kernel_info_t* name_to_kernel_infos;

#endif //_CPU_UTILS_H_
