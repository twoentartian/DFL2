//
// Created by tyd, 17-Sep-21.
//

/// how to use:
/// First of all, you need to have a working configuration file, this tool only process nodes information.
/// 1) ./large_scale_simulation_generator --> load generator config file to ../simulation/simulator_config.json
/// 2) ./large_scale_simulation_generator {path} --> load generator config file to certain path
/// ///

#include <iostream>
#include <algorithm>
#include <random>
#include <set>
#include <glog/logging.h>
#include <configure_file.hpp>

#include "../simulation/simulation_config_format.hpp"
#include "simulation_config_generator_common_functions.hpp"

configuration_file::json get_default_simulation_configuration()
{
	configuration_file::json output;
	
	output["node_count"] = 100;
	output["buffer_size"] = 100;
	output["dataset_mode"] = "iid";
	output["model_generation_type"] = "normal";
	output["filter_limit"] = 0.5;
	output["training_interval_tick"] = configuration_file::json::array({8,9,10,11,12});
	
	output["node_peer_connection_count"] = 8;
	output["node_peer_connection_type"] = "--";
    output["node_peer_connection_generating_strategy"] = "exact";
	configuration_file::json malicious_node;
	malicious_node["malicious_model_poisoning_random_model"] = 1;
	malicious_node["malicious_data_poisoning_shuffle_label"] = 1;
	output["special_node"] = malicious_node;
	return output;
}

int main(int argc, char* argv[])
{
	configuration_file generator_config;
	generator_config.SetDefaultConfiguration(get_default_simulation_configuration());
	auto load_config_rc = generator_config.LoadConfiguration("large_scale_config.json");
	if (load_config_rc < 0)
	{
		LOG(FATAL) << "cannot load large scale configuration file, wrong format?";
		return -1;
	}
	auto my_config_json = generator_config.get_json();
	
	int node_count = *generator_config.get<int>("node_count");
	int buffer_size = *generator_config.get<int>("buffer_size");
	std::string dataset_mode = *generator_config.get<std::string>("dataset_mode");
	std::string model_generation_type = *generator_config.get<std::string>("model_generation_type");
	float filter_limit = *generator_config.get<float>("filter_limit");
	int node_peer_connection_count = *generator_config.get<int>("node_peer_connection_count");
	std::string node_peer_connection_type = *generator_config.get<std::string>("node_peer_connection_type");
	auto special_node = my_config_json["special_node"];
    std::string node_peer_connection_generating_strategy = my_config_json["node_peer_connection_generating_strategy"];
	std::cout << "node_count: " << node_count << std::endl;
	std::cout << "buffer_size: " << buffer_size << std::endl;
	std::cout << "dataset_mode: " << dataset_mode << std::endl;
	std::cout << "model_generation_type: " << model_generation_type << std::endl;
	std::cout << "filter_limit: " << filter_limit << std::endl;
	std::cout << "node_peer_connection_count: " << node_peer_connection_count << std::endl;
    std::cout << "node_peer_connection_generating_strategy: " << node_peer_connection_generating_strategy << std::endl;
	
    if (node_peer_connection_count >= node_count )
	{
        LOG(FATAL) << "node_peer_connection_count must be smaller than node_count " << "(" << node_peer_connection_count << "<" << node_count << ")"<< std::endl;
		return -1;
	}
	
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
		for (int i = 0; i < node_count; ++i)
		{
			configuration_file::json node;
			node["name"] = std::to_string(i);
			node["dataset_mode"] = dataset_mode;
			node["training_interval_tick"] = my_config_json["training_interval_tick"] ;
			node["buffer_size"] = buffer_size;
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
        if (node_peer_connection_generating_strategy == "exact")
        {
            bool flop_connection = (node_peer_connection_count > node_count / 2) && (node_peer_connection_type == simulation_config_format::bilateral_term); //if true, peer = all nodes - peers
            int node_peer_connection_count_override = flop_connection?(node_count - 1 - node_peer_connection_count):node_peer_connection_count;
            LOG_IF(FATAL, node_peer_connection_count * node_count % 2 != 0) << "impossible situation: node_peer_connection_count("<< node_peer_connection_count << ") * node_count(" << node_count << ") % 2 != 0";

            int try_count = 0;
            bool whole_success = false;
            std::vector<std::tuple<int,int>> connections;
            
            while (try_count < 10000)
            {
                try_count++;
    
                std::map<int, int> peer_per_node;
                for (int node = 0; node < node_count; ++node)
                {
                    peer_per_node[node] = node_peer_connection_count_override;
                }
                
                auto connection_result = generate_network_topology(peer_per_node);
                if (connection_result.has_value()) //success
                {
                    connections = *connection_result;
                    
                    //check whether we get a network with islands
                    std::map<int, std::set<int>> peer_map;
                    for (int node = 0; node < node_count; ++node)
                    {
                        peer_map[node] = {};//there is a node but no peer
                    }
                    for (auto &[node, peer]: connections)
                    {
                        peer_map[node].emplace(peer);
                        peer_map[peer].emplace(node);//add peers
                    }
                    if (flop_connection)
                    {
                        std::set<int> whole_set;//should contain node 0...node_count-1
                        for (int node = 0; node < node_count; ++node)
                        {
                            whole_set.emplace(node);
                        }
                        for (auto& [node, all_peers]: peer_map)
                        {
                            std::set<int> flopped_connections;
                            std::set_difference(whole_set.begin(), whole_set.end(), all_peers.begin(), all_peers.end(), std::insert_iterator<std::set<int>>(flopped_connections, flopped_connections.begin()));
                            peer_map[node] = flopped_connections;
                            peer_map[node].erase(node);//remove self
                        }
                    }
                    
                    //check islands
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
                std::cout << "generate network after " << try_count << " tries(" << std::setprecision(4) << 100.0/try_count << "%)" << std::endl;
            }
            else
            {
                LOG(FATAL) << "cannot generate network after " << try_count << " tries" << std::endl;
            }

            auto &node_topology_json = config_json["node_topology"];
            node_topology_json.clear();
            if (flop_connection)
            {
                std::map<int, std::set<int>> real_connections;
                std::set<int> all_nodes;
                for (int node = 0; node < node_count; ++node)
                {
                    all_nodes.emplace(node);
                }
                for (int node = 0; node <node_count; ++node)
                {
                    real_connections[node] = all_nodes;
                    real_connections[node].erase(node);
                }
                for (auto &[node0, node1]: connections)
                {
                    real_connections[node0].erase(node1);
                }

                for (auto& [node, peers]: real_connections)
                {
                    for (auto& single_peer: peers)
                    {
                        if (node >= single_peer) continue;
                        std::cout << std::to_string(node) << node_peer_connection_type << std::to_string(single_peer) << std::endl;
                        node_topology_json.push_back(std::to_string(node) + node_peer_connection_type + std::to_string(single_peer));
                    }
                }
            }
            else
            {
                for (auto &[node0, node1]: connections)
                {
                    std::cout << std::to_string(node0) << node_peer_connection_type << std::to_string(node1) << std::endl;
                    node_topology_json.push_back(std::to_string(node0) + node_peer_connection_type + std::to_string(node1));
                }
            }
        }
        else if (node_peer_connection_generating_strategy == "rough")
        {
            std::vector<int> node_list_ex_self;
            node_list_ex_self.resize(node_count-1);
            for (int i = 0; i < node_list_ex_self.size(); ++i)
            {
                node_list_ex_self[i] = i;
            }
            std::vector<std::vector<int>> node_connections;
            node_connections.resize(node_count);
            static std::random_device rd;
            static std::mt19937 g(rd());
            for (int i = 0; i < node_count; ++i)
            {
                std::shuffle(node_list_ex_self.begin(), node_list_ex_self.end(), g);
                int count = node_peer_connection_count;
                if (node_peer_connection_type == simulation_config_format::bilateral_term)
                {
                    count = count / 2;
                }
                else if (node_peer_connection_type == simulation_config_format::unidirectional_term)
                {
                    count = count;
                }
                else
                {
                    LOG(FATAL) << "never reach";
                }

                auto iter = node_list_ex_self.begin();
                while (count --)
                {
                    node_connections[i].push_back(*iter);
                    iter++;
                }

                for (auto &peer : node_connections[i])
                {
                    if (peer >= i)
                    {
                        peer++;
                    }
                }
            }

            auto &node_topology_json = config_json["node_topology"];
            node_topology_json.clear();
            for (int i = 0; i < node_count; ++i)
            {
                for (auto& peer : node_connections[i])
                {
                    std::cout << std::to_string(i) << node_peer_connection_type << std::to_string(peer) << std::endl;
                    node_topology_json.push_back(std::to_string(i) + node_peer_connection_type + std::to_string(peer));
                }
            }
        }
        else
        {
            LOG(FATAL) << "never reach";
        }
		
		config_json["ml_delayed_test_accuracy"] = false;
        
        //write simulation config generator info to output config file.
        apply_generator_config_to_output_config(generator_config.get_json(), config_json, "comment_this_config_file_is_initially_generated_by_large_scale_simulation_generator_with_following_config", true);
        
		config.write_back();
	}
	
	return 0;
}
