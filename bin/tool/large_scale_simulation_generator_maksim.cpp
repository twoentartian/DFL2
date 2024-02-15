/// how to use:
/// First of all, you need to have a working configuration file, this tool only process nodes information.
/// 1) ./large_scale_simulation_generator --> load generator config file to ../simulation/simulator_config.json
/// 2) ./large_scale_simulation_generator {path} --> load generator config file to certain path
/// ///

#include <iostream>
#include <algorithm>
#include <random>
#include <set>
#include <cctype>
#include <glog/logging.h>
#include <configure_file.hpp>

#include "../simulation/simulation_config_format.hpp"
#include "simulation_config_generator_common_functions.hpp"

configuration_file::json get_default_simulation_configuration()
{
    configuration_file::json output;

    output["dataset_mode"] = "iid";
    output["model_generation_type"] = "normal";
    output["filter_limit"] = 0.5;
    output["training_interval_tick"] = configuration_file::json::array({8, 9, 10, 11, 12});

    output["social_network_buffer_to_peer_ratio"] = 1.0;
    output["node_peer_connection_type"] = "--";
    output["network_topology_maksim_format"] = "./1.conn.dat";
    configuration_file::json malicious_node;
    malicious_node["malicious_model_poisoning_random_model"] = 1;
    malicious_node["malicious_data_poisoning_shuffle_label"] = 1;
    output["special_node"] = malicious_node;
    return output;
}

int countIntegersInString(const std::string& str) {
    int count = 0;
    bool inNumber = false; // Flag to track if we're currently in a sequence of digits

    for (size_t i = 0; i < str.length(); ++i) {
        if (isdigit(str[i])) {
            if (!inNumber) {
                inNumber = true; // We've started a new number
                ++count;
            }
        } else {
            inNumber = false; // No longer in a sequence of digits
        }
    }
    return count;
}

int main(int argc, char* argv[])
{
	configuration_file generator_config;
	generator_config.SetDefaultConfiguration(get_default_simulation_configuration());
	auto load_config_rc = generator_config.LoadConfiguration("large_scale_config_maksim.json");
	if (load_config_rc < 0)
	{
		LOG(FATAL) << "cannot load large scale configuration file, wrong format?";
		return -1;
	}
	auto my_config_json = generator_config.get_json();


    std::string dataset_mode = *generator_config.get<std::string>("dataset_mode");
    std::string model_generation_type = *generator_config.get<std::string>("model_generation_type");
    float filter_limit = *generator_config.get<float>("filter_limit");
    float social_network_buffer_to_peer_ratio = *generator_config.get<float>("social_network_buffer_to_peer_ratio");
    std::string node_peer_connection_type = *generator_config.get<std::string>("node_peer_connection_type");
    auto special_node = my_config_json["special_node"];
    std::string network_topology_maksim_format = my_config_json["network_topology_maksim_format"];

    //load network_topology
    std::map<int, std::set<int>> total_nodes_and_peers;
    {
        std::ifstream input_file(network_topology_maksim_format);
        if (!input_file)
        {
            LOG(FATAL) << "topology file " << network_topology_maksim_format << " doesn't exist." << std::endl;
        }
        std::string line;
        while (std::getline(input_file, line))
        {
            std::istringstream iss(line);
            int integerCount = countIntegersInString(line);
            if (integerCount == 1) {
                int a;
                if (!(iss >> a)) {LOG(FATAL) << "the format is not a pair of int. Line: " << line;}
                if (!total_nodes_and_peers.contains(a)) {
                    std::set<int> temp;
                    total_nodes_and_peers[a] = temp;
                }
            }
            else if (integerCount == 2) {
                int a, b;
                if (!(iss >> a >> b)) {LOG(FATAL) << "the format is not a pair of int. Line: " << line;}
                total_nodes_and_peers[a].insert(b);
                total_nodes_and_peers[b].insert(a);
            }
            else {
                LOG(FATAL) << "unknown format: " << line;
            }
        }
    }

    size_t node_count = total_nodes_and_peers.size();
    std::cout << "node_count: " << node_count << std::endl;
    std::cout << "dataset_mode: " << dataset_mode << std::endl;
    std::cout << "model_generation_type: " << model_generation_type << std::endl;
    std::cout << "filter_limit: " << filter_limit << std::endl;
    std::cout << "social_network_buffer_to_peer_ratio: " << social_network_buffer_to_peer_ratio << std::endl;

	if (node_peer_connection_type != simulation_config_format::bilateral_term && node_peer_connection_type != simulation_config_format::unidirectional_term)
	{
        LOG(FATAL) << "unknown node_peer_connection_type: " << node_peer_connection_type << std::endl;
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
            std::cout <<  "how to use:" << std::endl;
            std::cout <<  "First of all, you need to have a working configuration file, this tool only process nodes information." << std::endl;
            std::cout <<  "1) ./large_scale_simulation_generator --> load generate config file to ../simulation/simulator_config.json" << std::endl;
            std::cout <<  "2) ./large_scale_simulation_generator {path} --> load generate config file to certain path" << std::endl;
			return -1;
		}
		
		std::filesystem::path config_file_path(config_file);
		if (!std::filesystem::exists(config_file_path))
		{
			std::cerr << config_file << " does not exist" << std::endl;
			return -1;
		}
		
		config.LoadConfiguration(config_file);
		auto& config_json = config.get_json();
		configuration_file::json& nodes_json = config.get_json()["nodes"];
		nodes_json.clear();
		for (const auto& [target_node, peers] : total_nodes_and_peers)
		{
			configuration_file::json node;
			node["name"] = std::to_string(target_node);
			node["dataset_mode"] = dataset_mode;
			node["training_interval_tick"] = my_config_json["training_interval_tick"];
            node["buffer_size"] = static_cast<int>(social_network_buffer_to_peer_ratio * static_cast<float>(peers.size()));
			node["model_generation_type"] = model_generation_type;
			node["filter_limit"] = filter_limit;
			node["node_type"] = "normal";
			configuration_file::json node_non_iid = configuration_file::json::object();
			node_non_iid["1"] = configuration_file::json::array({1.0,2.0});
			node_non_iid["3"] = configuration_file::json::array({2.0,3.0});
			node["non_iid_distribution"] = node_non_iid;
			nodes_json.push_back(node);
		}
		
		////add special nodes
		int node_index = 0;
		for (auto& [node_type, count] : special_node.items())
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
		
		////set node topology
		{
			auto &node_topology_json = config_json["node_topology"];
			node_topology_json.clear();
			std::set<std::tuple<int,int>> duplicate_connection_checker;
			for (const auto& [target_node, peers]: total_nodes_and_peers)
			{
				for (const auto &single_peer: peers)
				{
                    if (duplicate_connection_checker.contains({target_node, single_peer})) {
                        //skip
                    }
                    else {
                        duplicate_connection_checker.insert({target_node, single_peer});
                        duplicate_connection_checker.insert({single_peer, target_node});
                        std::cout << std::to_string(target_node) << node_peer_connection_type << std::to_string(single_peer) << std::endl;
                        node_topology_json.push_back(std::to_string(target_node) + node_peer_connection_type + std::to_string(single_peer));
                    }
				}
			}
		}
		
		config_json["ml_delayed_test_accuracy"] = false;
        
        //write simulation config generator info to output config file.
        apply_generator_config_to_output_config(generator_config.get_json(), config_json, "comment_this_config_file_is_initially_generated_by_large_scale_simulation_generator_with_following_config", true);
        
		config.write_back();
	}
	
	return 0;
}
