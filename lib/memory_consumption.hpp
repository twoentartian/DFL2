#pragma once

#include <regex>
#include <fstream>
#include <sstream>

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

size_t get_memory_consumption_byte(){ //Note: this value is in KB!
    size_t result = 0;
    
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line))
    {
        if (line.starts_with("VmSize"))
        {
            std::regex e ("\\d+");
            std::sregex_iterator iter(line.begin(), line.end(), e);
            std::sregex_iterator end;
            result = std::stoull((*iter)[0]);
        }
    }
    
    return result*1024;
}

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */

#endif


