#pragma once

#include <sstream>
#include <iostream>
#include <dfl_util.hpp>

void introducer_CLI_print(const std::stringstream& data)
{
    dfl_util::print_info_to_log_stdcout(data);
    std::cout << ">> ";
    std::cout.flush();
}