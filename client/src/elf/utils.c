#include "uthash.h"
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/wait.h>
#include <openssl/md5.h>
#include <linux/limits.h>
#include "rpc/types.h"
#include <sys/stat.h>

#include "utils.h"
#include "log.h"

kernel_info_t* name_to_kernel_infos = NULL;
kernel_info_t* func_ptr_to_kernel_infos = NULL;

kernel_info_t* find_kernel_host_func(const void* func) {
    kernel_info_t* info;
    HASH_FIND(hh_host_func, func_ptr_to_kernel_infos, &func, sizeof(void*), info);
    return info;
}
kernel_info_t* find_kernel_name(const char* name) {
    kernel_info_t* info;
    HASH_FIND(hh_name, name_to_kernel_infos, name, strlen(name), info);
    return info;
}
void add_kernel_host_func(const void* func, kernel_info_t* info) {
    if (find_kernel_host_func(func) == NULL) {
        HASH_ADD(hh_host_func, func_ptr_to_kernel_infos, host_fun, sizeof(void*), info);
    }
}
void add_kernel_name(const char* name, kernel_info_t* info) {
    if (find_kernel_name(name) == NULL) {
        // note the difference between `char*` and `char[]`
        HASH_ADD_KEYPTR(hh_name, name_to_kernel_infos, info->name, strlen(info->name), info);
    }
}

int cpu_utils_launch_child(const char *file, char **args)
{
    int filedes[2];
    FILE *fd = NULL;

    if (pipe(filedes) == -1) {
        LOGE(LOG_ERROR, "error while creating pipe");
        return -1;
    }

    pid_t pid = fork();
    if (pid == -1) {
        LOGE(LOG_ERROR, "error while forking");
        return -1;
    } else if (pid == 0) {
        while ((dup2(filedes[1], STDOUT_FILENO) == -1) && (errno == EINTR)) {}
        while ((dup2(filedes[1], STDERR_FILENO) == -1) && (errno == EINTR)) {}
        close(filedes[1]);
        close(filedes[0]);
        char *env[] = {NULL};
        execvpe(file, args, env);
        exit(1);
    }
    close(filedes[1]);
    return filedes[0];
}

kernel_info_t* utils_search_info(const char *kernelname)
{
    LOGE(LOG_DBG(1), "searching for %s in %d entries", kernelname, HASH_CNT(hh_name, name_to_kernel_infos));
    return find_kernel_name(kernelname);
}


static int cpu_utils_read_pars(kernel_info_t *info, FILE* fdesc)
{
    static const char* attr_str[] = {"EIATTR_KPARAM_INFO",
        "EIATTR_CBANK_PARAM_SIZE",
        "EIATTR_PARAM_CBANK"};
    enum attr_t {KPARAM_INFO = 0,
        CBANK_PARAM_SIZE = 1,
        PARAM_CBANK = 2,
        ATTR_T_LAST}; // states for state machine
    char *line = NULL;
    size_t linelen = 0;
    int ret = 1;
    int read = 0;
    char key[32];
    char val[256] = {0};
    size_t val_len = 0;
    enum attr_t cur_attr = ATTR_T_LAST; // current state of state machine
    int consecutive_empty_lines = 0;
    info->param_num = 0;
    info->param_offsets = NULL;
    info->param_sizes = NULL;
    while (getline(&line, &linelen, fdesc) != -1) {
        memset(val, 0, 256);
        read = sscanf(line, "%31s %255c\n", key, val);
        val_len = strlen(val);
        if (val_len > 0) {
            val[strlen(val)-1] = '\0';
        }
        if (read == -1 || read == 0) {
            if (++consecutive_empty_lines >= 2) {
                break; //two empty line means there is no more info for this kernel
            } else {
                continue;
            }
        } else {
            consecutive_empty_lines = 0;
            if (read == 1) {
                continue; // some lines have no key-value pair.
                // We are not interested in those lines.
            }
        }
        if (strcmp(key, "Attribute:") == 0) { // state change
            LOG(LOG_DBG(3), "\"%s\", \"%s\"", key, val);
            cur_attr = ATTR_T_LAST;
            for (int i=0; i < ATTR_T_LAST; i++) {
                if (strcmp(val, attr_str[i]) == 0) {
                    LOG(LOG_DBG(3), "found %s", attr_str[i]);
                    cur_attr = i;
                }
            }
        } else if(strcmp(key, "Value:") == 0) {
            LOG(LOG_DBG(3), "\"%s\", \"%s\"", key, val);
            size_t buf;
            uint16_t ordinal, offset, size;
            switch(cur_attr) {
            case KPARAM_INFO:
                if (sscanf(val, "Index : 0x%*hx Ordinal : 0x%hx Offset : 0x%hx Size : 0x%hx\n", &ordinal, &offset, &size) != 3 ) {
                    LOGE(LOG_ERROR, "unexpected format of cuobjdump output");
                    goto cleanup;
                }
                if (ordinal >= info->param_num) {
                    info->param_offsets = realloc(
                                                  info->param_offsets,
                                                  (ordinal+1)*sizeof(uint16_t));
                    info->param_sizes = realloc(
                                                info->param_sizes,
                                                (ordinal+1)*sizeof(uint16_t));
                    info->param_num = ordinal+1;
                }
                info->param_offsets[ordinal] = offset;
                info->param_sizes[ordinal] = size;
                break;
            case CBANK_PARAM_SIZE:
                if (sscanf(val, "0x%lx", &info->param_size) != 1) {
                    LOGE(LOG_ERROR, "value has wrong format: key: %s, val: %s", key, val);
                    goto cleanup;
                }
                break;
            case PARAM_CBANK:
                if (sscanf(val, "0x%*x 0x%lx", &buf) != 1) {
                    LOGE(LOG_ERROR, "value has wrong format: key: %s, val: %s", key, val);
                    goto cleanup;
                }
                LOG(LOG_DBG(3), "found param address: %d", (uint16_t)(buf & 0xFFFF));
                break;
            default:
                break;
            }
        }


    }

    ret = 0;
 cleanup:
    free(line);
    return ret;
}

int cpu_utils_contains_kernel(const char *path)
{
    int ret = 1;
    char linktarget[PATH_MAX] = {0};
    char *args[] = {"/usr/local/cuda/bin/cuobjdump", "--dump-elf", NULL, NULL};
    int output;
    FILE *fdesc; //fd to read subcommands output from
    int child_exit = 0;
    char *line = NULL;
    size_t linelen;
    static const char nv_info_prefix[] = ".nv.info.";
    kernel_info_t *buf = NULL;
    char *kernelname;
    struct stat filestat = {0};

    if (stat(path, &filestat) != 0) {
        LOGE(LOG_ERROR, "stat on %s failed.", path);
        goto out;
    }

    if (S_ISLNK(filestat.st_mode)) {
        if (readlink("/proc/self/exe", linktarget, PATH_MAX) == PATH_MAX) {
            LOGE(LOG_ERROR, "executable path length is too long");
            goto out;
        }
        args[2] = linktarget;
    } else {
        args[2] = (char*)path;
    }
    LOG(LOG_DBG(1), "searching for kernels in \"%s\".", args[2]);

    if ( (output = cpu_utils_launch_child(args[0], args)) == -1) {
        LOGE(LOG_ERROR, "error while launching child.");
        goto out;
    }

    if ( (fdesc = fdopen(output, "r")) == NULL) {
        LOGE(LOG_ERROR, "erro while opening stream");
        goto cleanup;
    }

    if (getline(&line, &linelen, fdesc) != -1) {
        /*if (strncmp(line, nv_info_prefix, strlen(nv_info_prefix)) != 0) {
            // Line does not start with .nv.info. so continue searching.
            continue;
        }*/
        line[strlen(line)-1] = '\0';
        LOGE(LOG_DEBUG, "output: \"%s\"", line);
    }
    ret = 0;
    fclose(fdesc);
 cleanup:
    close(output);
    wait(&child_exit);
    LOG(LOG_DBG(1), "child exit code: %d", child_exit);
 out:
    free(line);
    return ret == 0 && child_exit == 0;
}

int utils_parameter_info(char *path)
{
    int ret = 1;
    char linktarget[PATH_MAX] = {0};
    char *args[] = {"/usr/local/cuda/bin/cuobjdump", "--dump-elf", NULL, NULL};
    int output;
    FILE *fdesc; //fd to read subcommands output from
    int child_exit = 0;
    char *line = NULL;
    size_t linelen;
    static const char nv_info_prefix[] = ".nv.info.";
    kernel_info_t *buf = NULL;
    char *kernelname;
    struct stat filestat = {0};

    if (path == NULL) {
        LOGE(LOG_ERROR, "path is NULL.");
        goto out;
    }

    if (stat(path, &filestat) != 0) {
        LOGE(LOG_ERROR, "stat on %s failed.", path);
        goto out;
    }

    if (S_ISLNK(filestat.st_mode) || strcmp(path, "/proc/self/exe") == 0) {
        if (readlink("/proc/self/exe", linktarget, PATH_MAX) == PATH_MAX) {
            LOGE(LOG_ERROR, "executable path length is too long");
            goto out;
        }
        args[2] = linktarget;
    } else {
        args[2] = path;
    }
    LOG(LOG_DBG(1), "searching for kernels in \"%s\".", args[2]);

    if ( (output = cpu_utils_launch_child(args[0], args)) == -1) {
        LOGE(LOG_ERROR, "error while launching child.");
        goto out;
    }

    if ( (fdesc = fdopen(output, "r")) == NULL) {
        LOGE(LOG_ERROR, "erro while opening stream");
        goto cleanup1;
    }

    while (getline(&line, &linelen, fdesc) != -1) {
        if (strncmp(line, nv_info_prefix, strlen(nv_info_prefix)) != 0) {
            // Line does not start with .nv.info. so continue searching.
            continue;
        }
        // Line starts with .nv.info.
        // Kernelname is line + strlen(nv_info_prefix)
        kernelname = line + strlen(nv_info_prefix);
        if (strlen(kernelname) == 0) {
            LOGE(LOG_ERROR, "found .nv.info section, but kernelname is empty");
            goto cleanup2;
        }

        buf = (kernel_info_t*) malloc(sizeof(*buf));

        size_t buflen = strlen(kernelname);
        if ((buf->name = malloc(buflen)) == NULL) {
            LOGE(LOG_ERROR, "malloc failed");
            goto cleanup2;
        }
        //copy string and remove trailing \n
        strncpy(buf->name, kernelname, buflen-1);
        buf->name[buflen-1] = '\0';

        if (cpu_utils_read_pars(buf, fdesc) != 0) {
            LOGE(LOG_ERROR, "reading paramter infos failed.\n");
            goto cleanup2;
        }
        add_kernel_name(buf->name, buf);
        LOG(LOG_DEBUG, "found kernel \"%s\" [param_num: %d, param_size: %d]",
            buf->name, buf->param_num, buf->param_size);

    }

    if (ferror(fdesc) != 0) {
        LOGE(LOG_ERROR, "file descriptor shows an error");
        goto cleanup2;
    }

    ret = 0;
 cleanup2:
    fclose(fdesc);
 cleanup1:
    close(output);
    wait(&child_exit);
    LOG(LOG_DBG(1), "child exit code: %d", child_exit);
 out:
    free(line);
    return ret == 0 && child_exit == 0;
}

void kernel_infos_free()
{
    kernel_info_t *info, *tmp;
    HASH_ITER(hh_host_func, func_ptr_to_kernel_infos, info, tmp) {
        HASH_DELETE(hh_host_func, func_ptr_to_kernel_infos, info);
    }
    HASH_ITER(hh_name, name_to_kernel_infos, info, tmp) {
        HASH_DELETE(hh_name, name_to_kernel_infos, info);
        free(info->name);
        free(info->param_offsets);
        free(info->param_sizes);
        free(info);
    }
}
