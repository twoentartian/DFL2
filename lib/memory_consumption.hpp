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
    extern "C" {
#include "sys/types.h"
#include "sys/sysinfo.h"
    size_t get_memory_consumption_byte() {
        struct sysinfo sys_info;
        if (sysinfo(&sys_info) != -1) {
            size_t totalVirtualMem = sys_info.totalram;
            totalVirtualMem += sys_info.totalswap;
            totalVirtualMem *= sys_info.mem_unit;
            return totalVirtualMem;
        }
        return 0;
    }
}

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */

#endif


