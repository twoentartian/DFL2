//
// Created by tyd on 30-9-22.
//

#include <filesystem>
#include <iostream>

#include <glog/logging.h>

#include "./large_scale_DFL_deployment_default_configuration.hpp"

constexpr char INTRODUCER_LOG_PATH[] = "./DFL_deployment_log/";

int main(int argc, char **argv)
{
    //log file path
    google::InitGoogleLogging(argv[0]);
    std::filesystem::path log_path(INTRODUCER_LOG_PATH);
    if (!std::filesystem::exists(log_path)) std::filesystem::create_directories(log_path);
    google::SetLogDestination(google::INFO, log_path.c_str());
    google::SetStderrLogging(google::WARNING);

    //load configuration
    configuration_file config;
    config.SetDefaultConfiguration(get_default_DFL_deployment_configuration());
    auto return_code = config.LoadConfiguration("./large_scale_DFL_deployment_config.json");
    if(return_code < configuration_file::NoError)
    {
        if (return_code == configuration_file::FileFormatError)
            LOG(FATAL) << "configuration file format error";
        else
            LOG(FATAL) << "configuration file error code: " << return_code;
    }

    configuration_file simulation_config;
    std::string simulation_config_path;
    if (argc == 2)
    {
        simulation_config_path.assign(argv[1]);
    }
    else if (argc == 1)
    {
        simulation_config_path.assign("../simulation/simulator_config.json");
    }
    else
    {
        std::cout <<  "how to use:" << std::endl;
        std::cout <<  "First of all, you need to have a simulation configuration file, this tool will convert the simulation config to a DFL deployment directory." << std::endl;
        std::cout <<  "1) ./large_scale_DFL_deployment --> load generate config file to ../simulation/simulator_config.json" << std::endl;
        std::cout <<  "2) ./large_scale_DFL_deployment {path} --> load generate config file to certain path" << std::endl;
        return -1;
    }

    std::filesystem::path config_file_path(simulation_config_path);
    if (!std::filesystem::exists(config_file_path))
    {
        std::cerr << simulation_config_path << " does not exist" << std::endl;
        return -1;
    }

    config.LoadConfiguration(simulation_config_path);
    auto [status, msg] = check_config(config);
    if (!status)
    {
        std::cerr << msg << std::endl;
        return -1;
    }



    return 0;
}