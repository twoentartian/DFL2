#pragma once

#include <string>
#include <tuple>
#include <filesystem>

#include <configure_file.hpp>

configuration_file::json get_default_DFL_deployment_configuration()
{
    configuration_file::json output;

    output["path_exe_DFL"] = "../DFL/DFL";
    output["path_exe_injector"] = "../data_injector/data_injector_mnist";
    output["path_exe_introducer"] = "../DFL_introducer/DFL_introducer";
    output["path_dll_reputation"] = "../reputation_sdk/sample/libreputation_api_sample.so";



    return output;
}

std::tuple<bool, std::string> check_config(const configuration_file& config_json)
{
    const std::vector<std::string> path_check = {"path_exe_DFL", "path_exe_injector", "path_exe_introducer", "path_dll_reputation"};

    for (auto&& item: path_check) {
        auto exe_path_opt = config_json.get<std::string>("path_exe_DFL");
        if (!exe_path_opt) return {false, "configuration {" + item + "} is empty"};
        std::filesystem::path exe_path(*exe_path_opt);
        if (!std::filesystem::exists(exe_path))
        {
            return {false, "configuration {" + item + "}, path{" + *exe_path_opt + "} doesn't exist"};
        }
    }

    return {true, ""};
}