//
// Created by tyd, 17-Sep-21.
//

/// how to use:
/// First of all, you need to have a working configuration file, this tool only process nodes information.
/// 1) ./large_scale_simulation_generator --> load generate config file to ../simulation/simulator_config.json
/// 2) ./large_scale_simulation_generator {path} --> load generate config file to certain path
/// ///

#include <iostream>
#include <algorithm>
#include <random>
#include <set>
#include <glog/logging.h>
#include <configure_file.hpp>

#include "../simulation/simulation_config_format.hpp"

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
	configuration_file my_config;
	my_config.SetDefaultConfiguration(get_default_simulation_configuration());
	auto load_config_rc = my_config.LoadConfiguration("large_scale_config.json");
	if (load_config_rc < 0)
	{
		LOG(FATAL) << "cannot load large scale configuration file, wrong format?";
		return -1;
	}
	auto my_config_json = my_config.get_json();
	
	int node_count = *my_config.get<int>("node_count");
	int buffer_size = *my_config.get<int>("buffer_size");
	std::string dataset_mode = *my_config.get<std::string>("dataset_mode");
	std::string model_generation_type = *my_config.get<std::string>("model_generation_type");
	float filter_limit = *my_config.get<float>("filter_limit");
	int node_peer_connection_count = *my_config.get<int>("node_peer_connection_count");
	std::string node_peer_connection_type = *my_config.get<std::string>("node_peer_connection_type");
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
		
		//add malicious nodes
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

            int try_count = 0;
            bool whole_success = false;
            std::vector<std::tuple<int,int>> connections;

            while (try_count < 10000)
            {
                try_count++;
                bool success = true;
                std::random_device rd;
                std::mt19937 g(rd());

                //init variables
                connections.clear();
                std::map<int, int> node_instances_counter;
                std::map<int, std::set<int>> node_ban_list;
                std::map<int, std::set<int>> node_available_nodes;
                std::set<int> all_nodes;
                for (int node = 0; node < node_count; ++node)
                {
                    all_nodes.emplace(node);
                }
                for (int node = 0; node < node_count; ++node)
                {
                    node_instances_counter[node] = node_peer_connection_count_override;
                    node_available_nodes[node] = all_nodes;
                    if (node_peer_connection_type == simulation_config_format::bilateral_term)
                    {
                        for (int node_to_remove = 0; node_to_remove < node; ++node_to_remove)
                        {
                            node_available_nodes[node].erase(node_to_remove);
                        }
                    }
                }

                //try to generate network
                for (auto& [node_name, instance] : node_instances_counter)
                {
                    if (!success) break;

                    while (node_instances_counter[node_name] != 0)
                    {
                        if (node_available_nodes[node_name].empty())
                        {
                            //we should retry
                            success = false;
                            break;
                        }

                        std::uniform_int_distribution dist(0, int(node_available_nodes[node_name].size())-1);
                        auto it = std::begin(node_available_nodes[node_name]);
                        std::advance(it,dist(g));
                        auto random_pick_node = *it;
                        node_available_nodes[node_name].erase(it);//remove this picked node
                        if (random_pick_node == node_name) continue;
                        if (node_ban_list[node_name].contains(random_pick_node)) continue;
                        if (node_instances_counter[random_pick_node] == 0) continue;

                        if (node_peer_connection_type == simulation_config_format::bilateral_term)
                        {
                            node_ban_list[node_name].emplace(random_pick_node);
                            node_ban_list[random_pick_node].emplace(node_name);
                        }
                        else if (node_peer_connection_type == simulation_config_format::unidirectional_term)
                        {
                            node_ban_list[node_name].emplace(random_pick_node);
                        }
                        else
                        {
                            LOG(FATAL) << "never reach";
                        }

                        node_instances_counter[node_name]--;
                        node_instances_counter[random_pick_node]--;
                        connections.emplace_back(node_name, random_pick_node);
                    }
                }
                if (success)
                {
                    whole_success = true;
                    break;
                }
            }

            if (whole_success)
            {
                std::cout << "generate network after " << try_count << " tries" << std::endl;
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
            std::random_device rd;
            std::mt19937 g(rd());
            for (int i = 0; i < node_count; ++i)
            {
                std::shuffle(node_list_ex_self.begin(), node_list_ex_self.end(), g);
                int count = node_peer_connection_count;
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
                std::cout << std::to_string(i) << node_peer_connection_type << "{";
                if (node_peer_connection_type == simulation_config_format::bilateral_term)
                {
                    for (int j = 0; j < node_peer_connection_count / 2; ++j)
                    {
                        std::cout << std::to_string(node_connections[i][j]) << ", ";
                        node_topology_json.push_back(std::to_string(i) + node_peer_connection_type + std::to_string(node_connections[i][j]));
                    }
                }
                else if (node_peer_connection_type == simulation_config_format::unidirectional_term)
                {
                    for (int j = 0; j < node_peer_connection_count; ++j)
                    {
                        std::cout << std::to_string(node_connections[i][j]) << ", ";
                        node_topology_json.push_back(std::to_string(i) + node_peer_connection_type + std::to_string(node_connections[i][j]));
                    }
                }
                else
                {
                    LOG(FATAL) << "never reach";
                }
                std::cout << "}" << std::endl;
            }
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
