//
// Created by tyd, 25-Feb-23.
//

/// how to use:
/// First of all, you need to have a working configuration file, this tool only process nodes information.
/// 1) ./large_scale_simulation_generator_social_network --> load generator config file to ../simulation/simulator_config.json
/// 2) ./large_scale_simulation_generator_social_network {path} --> load generator config file to certain path
/// ///

#include <iostream>
#include <algorithm>
#include <random>
#include <set>
#include <glog/logging.h>
#include <configure_file.hpp>

#include "../simulation/simulation_config_format.hpp"
#include "large_scale_simulation_generator_common_functions.hpp"

configuration_file::json get_default_simulation_configuration()
{
    configuration_file::json output;
    
    output["node_count"] = 100;
//    output["buffer_size"] = 100;
    output["dataset_mode"] = "iid";
    output["model_generation_type"] = "normal";
    output["filter_limit"] = 0.5;
    output["training_interval_tick"] = configuration_file::json::array({8, 9, 10, 11, 12});
    
//    output["node_peer_connection_count"] = 8;
    output["social_network_node_peer_connection_gamma"] = 3.0;
    output["social_network_node_peer_connection_min_peer"] = 2;
    output["social_network_buffer_to_peer_ratio"] = 1.0;
    output["node_peer_connection_type"] = "--";
    output["node_peer_connection_generating_strategy"] = "exact";
    configuration_file::json malicious_node;
    malicious_node["malicious_model_poisoning_random_model"] = 1;
    malicious_node["malicious_data_poisoning_shuffle_label"] = 1;
    output["special_node"] = malicious_node;
    return output;
}

int main(int argc, char *argv[])
{
    configuration_file my_config;
    my_config.SetDefaultConfiguration(get_default_simulation_configuration());
    auto load_config_rc = my_config.LoadConfiguration("large_scale_config_social_network.json");
    if (load_config_rc < 0)
    {
        LOG(FATAL) << "cannot load large scale configuration file, wrong format?";
        return -1;
    }
    auto my_config_json = my_config.get_json();
    
    int node_count = *my_config.get<int>("node_count");
//    int buffer_size = *my_config.get<int>("buffer_size");
    std::string dataset_mode = *my_config.get<std::string>("dataset_mode");
    std::string model_generation_type = *my_config.get<std::string>("model_generation_type");
    float filter_limit = *my_config.get<float>("filter_limit");
    float social_network_node_peer_connection_gamma = *my_config.get<float>("social_network_node_peer_connection_gamma");
    int social_network_node_peer_connection_min_peer = *my_config.get<int>("social_network_node_peer_connection_min_peer");
    float social_network_buffer_to_peer_ratio = *my_config.get<float>("social_network_buffer_to_peer_ratio");
    std::string node_peer_connection_type = *my_config.get<std::string>("node_peer_connection_type");
    auto special_node = my_config_json["special_node"];
    std::string node_peer_connection_generating_strategy = my_config_json["node_peer_connection_generating_strategy"];
    std::cout << "node_count: " << node_count << std::endl;
//    std::cout << "buffer_size: " << buffer_size << std::endl;
    std::cout << "dataset_mode: " << dataset_mode << std::endl;
    std::cout << "model_generation_type: " << model_generation_type << std::endl;
    std::cout << "filter_limit: " << filter_limit << std::endl;
    std::cout << "social_network_node_peer_connection_gamma: " << social_network_node_peer_connection_gamma << std::endl;
    std::cout << "social_network_node_peer_connection_min_peer: " << social_network_node_peer_connection_min_peer << std::endl;
    std::cout << "social_network_buffer_to_peer_ratio: " << social_network_buffer_to_peer_ratio << std::endl;
    std::cout << "node_peer_connection_generating_strategy: " << node_peer_connection_generating_strategy << std::endl;
    
    if (node_peer_connection_type != simulation_config_format::bilateral_term && node_peer_connection_type != simulation_config_format::unidirectional_term)
    {
        LOG(FATAL) << "unknown node_peer_connection_type: " << node_peer_connection_type << std::endl;
        return -1;
    }
    
    if (node_peer_connection_generating_strategy != "exact" && node_peer_connection_generating_strategy != "rough")
    {
        LOG(FATAL) << R"(node_peer_connection_generating_strategy must be "exact" or "rough", get()" << node_peer_connection_generating_strategy << ")" << std::endl;
        return -1;
    }
    
    {
        ////set nodes
        configuration_file config;
        std::string config_file;
        
        if (argc == 2)
        {
            config_file.assign(argv[1]);
        }
        else if (argc == 1)
        {
            config_file.assign("../simulation/simulator_config.json");
        }
        else
        {
            std::cout << "how to use:" << std::endl;
            std::cout << "First of all, you need to have a working configuration file, this tool only process nodes information." << std::endl;
            std::cout << "1) ./large_scale_simulation_generator --> load generate config file to ../simulation/simulator_config.json" << std::endl;
            std::cout << "2) ./large_scale_simulation_generator {path} --> load generate config file to certain path" << std::endl;
            return -1;
        }
        
        std::filesystem::path config_file_path(config_file);
        if (!std::filesystem::exists(config_file_path))
        {
            std::cerr << config_file << " does not exist" << std::endl;
            return -1;
        }
        
        config.LoadConfiguration(config_file);
        auto &config_json = config.get_json();
        configuration_file::json &nodes_json = config.get_json()["nodes"];
        nodes_json.clear();
        for (int i = 0; i < node_count; ++i)
        {
            configuration_file::json node;
            node["name"] = std::to_string(i);
            node["dataset_mode"] = dataset_mode;
            node["training_interval_tick"] = my_config_json["training_interval_tick"];
//            node["buffer_size"] = buffer_size;
            node["model_generation_type"] = model_generation_type;
            node["filter_limit"] = filter_limit;
            node["node_type"] = "normal";
            configuration_file::json node_non_iid = configuration_file::json::object();
            node_non_iid["1"] = configuration_file::json::array({1.0, 2.0});
            node_non_iid["3"] = configuration_file::json::array({2.0, 3.0});
            node["non_iid_distribution"] = node_non_iid;
            nodes_json.push_back(node);
        }
        
        ////add malicious nodes
        {
            int node_index = 0;
            for (auto &[node_type, count]: special_node.items())
            {
                int node_index_per_type = 0;
                while (node_index_per_type < count)
                {
                    nodes_json[node_index]["node_type"] = node_type;
                    std::cout << "set node " << node_index << " to special: " << node_type << std::endl;
                    node_index_per_type++;
                    node_index++;
                }
            }
        }
        
        ////set node topology
        if (node_peer_connection_generating_strategy == "exact")
        {
            std::map<int, int> peer_per_node;
            
            ////Scale-free network: https://en.wikipedia.org/wiki/Scale-free_network
            ////P(k) = A k ^ (-gamma)
            std::map<int, double> weight_per_k;
            double total_weight = 0.0;
            for (int k = social_network_node_peer_connection_min_peer; k < node_count - 1; ++k) // from peer=1 to peer=(node-1)
            {
                double current_weight = std::pow(k, -social_network_node_peer_connection_gamma);
                weight_per_k[k] = current_weight;
                total_weight += current_weight;
            }
            for (auto& [k,weight]: weight_per_k)
            {
                weight /= total_weight;
            }
            
            std::vector<double> boundaries;
            boundaries.reserve(node_count);
            double accumulate_boundary = 0.0;
            for (auto &[k, weight]: weight_per_k)
            {
                boundaries.push_back(accumulate_boundary);
                accumulate_boundary += weight;
            }
            boundaries.push_back(accumulate_boundary);
    
            ////generate degree for each node
            std::vector<std::tuple<int, int>> connections;
            int generate_degree_count = 1;
            while (generate_degree_count++)
            {
                static std::random_device rd;
                static std::mt19937 engine(rd());
                std::uniform_real_distribution<double> distribution(0.0, 1.0);
                while (true)
                {
                    for (int node = 0; node < node_count; ++node)
                    {
                        double random_number = distribution(engine);
                        int select_degree = 0;
                        for (int i = 0; i < boundaries.size() - 1; ++i)
                        {
                            if (boundaries[i] <= random_number && random_number < boundaries[i + 1])
                            {
                                select_degree = i + social_network_node_peer_connection_min_peer;
                            }
                        }
                        peer_per_node[node] = select_degree;
                    }
        
                    ////check if it is possible
                    size_t total_connection_terminals = 0;
                    for (const auto &[node, peer_count]: peer_per_node)
                    {
                        total_connection_terminals += peer_count;
                    }
                    if (total_connection_terminals % 2 == 0)
                        break;
                }
    
                ////begin generating the network topology
                int try_count = 0;
                bool whole_success = false;
                while (try_count < 10000)
                {
                    try_count++;
        
                    auto connection_optional = generate_network_topology(node_count, peer_per_node);
                    if (connection_optional.has_value()) //success
                    {
                        connections = *connection_optional;
                        
                        ////check whether we get a network with islands
                        std::map<int, std::set<int>> peer_map;
                        for (auto &[node, peer]: connections)
                        {
                            peer_map[node].emplace(peer);
                            peer_map[peer].emplace(node);
                        }
                        
                        ////check islands
                        std::set<int> mainland;
                        add_to_mainland(peer_map.begin()->first, peer_map, mainland);
                        if (mainland.size() == peer_map.size())
                        {
                            whole_success = true;
                            break;
                        }
                    }
                }
    
                if (whole_success)
                {
                    std::cout << "generate network topology after " << try_count << " tries(" << std::setprecision(4) << 100.0 / try_count << "%)" << std::endl;
                    break;
                }
                else
                {
                    LOG(WARNING) << "cannot generate network after " << try_count << " tries, re-generating the node degrees." << std::endl;
                }
            }
            
            ////print number of peers
            for (const auto &[node, peer_count]: peer_per_node)
            {
                std::cout << "node " << node << " should have " << peer_count << " peers" << std::endl;
            }
            
            ////write model buffer size back to configuration file
            for (const auto &[node, peer_count]: peer_per_node)
            {
                nodes_json[node]["buffer_size"] = static_cast<int>(social_network_buffer_to_peer_ratio * static_cast<float>(peer_count));
            }
            
            auto &node_topology_json = config_json["node_topology"];
            node_topology_json.clear();
            
            for (auto &[node0, node1]: connections)
            {
                std::cout << std::to_string(node0) << node_peer_connection_type << std::to_string(node1) << std::endl;
                node_topology_json.push_back(std::to_string(node0) + node_peer_connection_type + std::to_string(node1));
            }
        }
        else if (node_peer_connection_generating_strategy == "rough")
        {
            LOG(FATAL) << "social network generator does not support \"rough\" generate type.";
        }
        else
        {
            LOG(FATAL) << "never reach";
        }
        
        config_json["ml_delayed_test_accuracy"] = false;
        config.write_back();
    }
    
    return 0;
}
