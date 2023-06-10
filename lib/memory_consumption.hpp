#pragma once

#include <iostream>

#if defined(_WIN32)
/* Windows -------------------------------------------------- */
//not tested yet
#include "windows.h"
#include "psapi.h"
size_t get_memory_consumption_byte() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    size_t virtualMemUsedByMe = pmc.PrivateUsage;
    return virtualMemUsedByMe;
}
    
#elif defined(__APPLE__) && defined(__MACH__)
/* OSX ------------------------------------------------------ */
//not tested yet
#include<mach/mach.h>
size_t get_memory_consumption_byte() {
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count))
    {
        return -1;
    }
    return t_info.virtual_size;
}
    
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
/* Linux ---------------------------------------------------- */
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}

size_t get_memory_consumption_byte(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];
    
    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result*1024;
}

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */

#endif


