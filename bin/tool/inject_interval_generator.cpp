#include <iostream>
#include <filesystem>
#include <random>

#include <glog/logging.h>

#include <configure_file.hpp>

#include "simulation_config_generator_common_functions.hpp"

int main(int argc, char* argv[])
{
    configuration_file config;
    std::string config_file;

    double stddev = 0, mean = 0, floating = 0;
    if (argc == 5)
    {
        config_file.assign(argv[1]);
        mean = std::stod(argv[2]);
        stddev = std::stod(argv[3]);
        floating = std::stod(argv[4]);
    }
    else if (argc == 4)
    {
        config_file.assign("../simulation/simulator_config.json");
        mean = std::stod(argv[1]);
        stddev = std::stod(argv[2]);
        floating = std::stod(argv[3]);
    }
    else
    {
        std::cout << "inject_interval_generator [config_file_path] [mean] [stddev] [floating]" << std::endl;
        std::cout << "inject_interval_generator [mean] [stddev] [floating]" << std::endl;
        return -1;
    }
    std::cout << "exp=" << mean << std::endl;
    std::cout << "div=" << stddev << std::endl;
    std::cout << "floating=" << floating << std::endl;

    std::filesystem::path config_file_path(config_file);
    if (!std::filesystem::exists(config_file_path))
    {
        std::cerr << config_file << " does not exist" << std::endl;
        return -1;
    }

    config.LoadConfiguration(config_file);

    //nodes
    static std::random_device rd;
    std::mt19937 gen{rd()};
    std::normal_distribution<double> normal_dis(mean, stddev);
    configuration_file::json& nodes_json = config.get_json()["nodes"];
    CHECK(nodes_json.is_array());

    for (auto& node : nodes_json)
    {
        auto interval_ticks = configuration_file::json::array();
        auto middle = normal_dis(gen);
        auto lower = static_cast<int>(std::round(middle - floating)), upper = static_cast<int>(std::round(middle+floating));
        if (lower < 1) lower = 1;
        if (upper <= lower) upper = lower + 1;
        for (int i = lower; i < upper; ++i)
        {
            interval_ticks.push_back(i);
        }
        node["training_interval_tick"] = interval_ticks;
        std::cout << node["name"] << ": " << interval_ticks.dump() << std::endl;
    }
    
    configuration_file::json generator_comment;
    generator_comment["inject_interval_mean"] = mean;
    generator_comment["inject_interval_stddev"] = stddev;
    generator_comment["inject_interval_floating"] = floating;
    apply_generator_config_to_output_config(generator_comment, config.get_json(), "comment_this_config_file_is_modified_by_inject_interval_generator", false);

    std::cout << "writing back" << std::endl;
    config.write_back();
    return 0;
}