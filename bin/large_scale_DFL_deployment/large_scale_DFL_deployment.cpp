//
// Created by tyd on 30-9-22.
//

#include <filesystem>
#include <iostream>
#include <algorithm>
#include <numeric>

#include <glog/logging.h>

#include <node.hpp>
#include <crypto.hpp>
#include <time_util.hpp>
#include <utility>

#include "../env.hpp"
#include "../simulation/simulation_config_format.hpp"

#include "large_scale_DFL_deployment_default_configuration.hpp"
#include "compile_time_content.hpp"

constexpr char DEPLOY_LOG_PATH[] = "./DFL_deployment_log/";

class introducer_node
{
public:
	std::string blockchain_address;
	std::string blockchain_public_key;
	std::string blockchain_private_key;
	
	uint16_t port;
	std::string ip;
	
	introducer_node(std::string ip, uint16_t port): port(port), ip(std::move(ip))
	{
		set_default_values();
	}
	
	void set_default_values()
	{
		//blockchain data
		auto [pubKey, priKey] = crypto::ecdsa_openssl::generate_key_pairs();
		auto pubKey_hex = pubKey.getHexMemory();
		auto address = crypto::sha256::digest_s(pubKey_hex.data(), pubKey_hex.size());
		
		blockchain_public_key = pubKey.getTextStr_lowercase();
		blockchain_private_key = priKey.getTextStr_lowercase();
		blockchain_address = address.getTextStr_lowercase();
	}
	
	[[nodiscard]] configuration_file::json generate_json_for_node() const
	{
		configuration_file::json output;
		output["address"] = blockchain_address;
		output["public_key"] = blockchain_public_key;
		output["ip"] = ip;
		output["port"] = port;
		return output;
	}
	
	[[nodiscard]] configuration_file::json generate_json_for_introducer() const
	{
		configuration_file::json output;
		output["blockchain_address"] = blockchain_address;
		output["blockchain_private_key"] = blockchain_private_key;
		output["blockchain_public_key"] = blockchain_public_key;
		output["port"] = port;
		return output;
	}
};

class node_deploy_info
{
public:
	int blockchain_estimated_block_size{};
	int data_storage_trigger_training_size{};
	int ml_test_batch_size{};
	
	std::string blockchain_address;
	std::string blockchain_public_key;
	std::string blockchain_private_key;
	
	std::string data_storage_db_path;
	std::string transaction_db_path;
	std::string blockchain_block_db_path;
	std::string ml_solver_proto_path;
	
	int data_storage_service_concurrency{};
	uint16_t data_storage_service_port{};
	
	std::string dataset_label_path;
	std::string dataset_path;
	
	std::string ml_model_stream_type;
	float ml_model_stream_compressed_filter_limit{};
	
	std::string reputation_dll_path;
	std::string reputation_dll_datatype;
	
	int timeout_second{};
	int transaction_count_per_model_update{};
	bool enable_profiler{};
	int buffer_size{};
	std::string name;
	dataset_mode_type dataset_mode;
	node_type node_malicious_type;
	std::unordered_map<int, std::tuple<float, float>> special_non_iid_distribution;
	
	//network
	uint16_t port;
	std::vector<introducer_node> introducer_nodes;
	bool use_preferred_peers_only{};
	int inactive_peer_second{};
	std::vector<std::string> preferred_peers;
	int maximum_peer;
	
	//injector
	int data_injector_inject_amount{};
	int data_injector_inject_interval_scale_ms_to_tick{};
	float data_injector_inject_interval_tick{};
	
public:
	configuration_file::json generate_dfl_node_config() const
	{
		configuration_file::json output;
		
		output["blockchain_estimated_block_size"] = blockchain_estimated_block_size;
		output["data_storage_trigger_training_size"] = data_storage_trigger_training_size;
		output["ml_test_batch_size"] = ml_test_batch_size;
		
		output["blockchain_address"] = blockchain_address;
		output["blockchain_public_key"] = blockchain_public_key;
		output["blockchain_private_key"] = blockchain_private_key;
		
		output["data_storage_db_path"] = data_storage_db_path;
		output["transaction_db_path"] = transaction_db_path;
		output["blockchain_block_db_path"] = blockchain_block_db_path;
		output["ml_solver_proto_path"] = ml_solver_proto_path;
		
		output["data_storage_service_concurrency"] = data_storage_service_concurrency;
		output["data_storage_service_port"] = data_storage_service_port;
		
		output["ml_model_stream_type"] = ml_model_stream_type;
		output["ml_model_stream_compressed_filter_limit"] = ml_model_stream_compressed_filter_limit;
		
		output["reputation_dll_path"] = reputation_dll_path;
		output["reputation_dll_datatype"] = reputation_dll_datatype;
		
		output["timeout_second"] = timeout_second;
		output["transaction_count_per_model_update"] = transaction_count_per_model_update;
		output["enable_profiler"] = enable_profiler;
		
		configuration_file::json introducers = configuration_file::json::array();
		for (const auto& single_introducer: introducer_nodes)
		{
			introducers.push_back(single_introducer.generate_json_for_node());
		}
		output["network"]["introducers"] = introducers;
		output["network"]["maximum_peer"] = maximum_peer;
		output["network"]["inactive_peer_second"] = inactive_peer_second;
		output["network"]["use_preferred_peers_only"] = use_preferred_peers_only;
		output["network"]["port"] = port;
		output["network"]["preferred_peers"] = preferred_peers;
		return output;
	}
	
	configuration_file::json generate_dfl_node_data_injector_config() const
	{
		configuration_file::json output;
		
		{
			std::filesystem::path temp(dataset_label_path);
			output["dataset_label_path"] = temp.filename();
		}
		{
			std::filesystem::path temp(dataset_path);
			output["dataset_path"] = temp.filename();
		}
		output["dataset_mode"] = generate_dataset_mode_type();
		
		output["ip_address"] = "127.0.0.1";
		output["ip_port"] = data_storage_service_port;
		output["inject_amount"] = data_injector_inject_amount;
		output["inject_interval_ms"] = static_cast<int>((data_injector_inject_interval_tick * static_cast<float>(data_injector_inject_interval_scale_ms_to_tick)));
		
		configuration_file::json node_non_iid = configuration_file::json::object();
		for (const auto& single_non_iid_item: special_non_iid_distribution)
		{
			node_non_iid[std::to_string(single_non_iid_item.first)] = configuration_file::json::array({std::get<0>(single_non_iid_item.second), std::get<1>(single_non_iid_item.second)});
		}
		output["non_iid_distribution"] = node_non_iid;
		
		return output;
	}
	
	void set_default_values()
	{
		//blockchain keys
		{
			auto [pubKey, priKey] = crypto::ecdsa_openssl::generate_key_pairs();
			auto pubKey_hex = pubKey.getHexMemory();
			auto address = crypto::sha256::digest_s(pubKey_hex.data(), pubKey_hex.size());
			
			blockchain_public_key = pubKey.getTextStr_lowercase();
			blockchain_private_key = priKey.getTextStr_lowercase();
			blockchain_address = address.getTextStr_lowercase();
		}
		
		//database path
		{
			data_storage_db_path = "./dataset_db";
			transaction_db_path = "./transaction_db";
			blockchain_block_db_path = "./blocks";
		}
		
		ml_solver_proto_path = compile_time_content::lenet_solver_memory_name;
	}
	
	void apply_deployment_information(const configuration_file& json)
	{
		dataset_label_path = *json.get<std::string>("path_mnist_dataset_label");
		dataset_path = *json.get<std::string>("path_mnist_dataset_data");
		
		std::filesystem::path reputation_dll_path_(*json.get<std::string>("path_dll_reputation"));
		reputation_dll_path = reputation_dll_path_.filename();
		reputation_dll_datatype = *json.get<std::string>("reputation_dll_datatype");
		
		data_storage_service_concurrency = *json.get<int>("data_storage_service_concurrency");
		ml_test_batch_size = *json.get<int>("ml_test_batch_size");
		timeout_second = *json.get<int>("timeout_second");
		inactive_peer_second = *json.get<int>("network_inactive_peer_second");
		maximum_peer = *json.get<int>("network_maximum_peer");
		
		blockchain_estimated_block_size = *json.get<int>("blockchain_estimated_block_size");
		data_storage_trigger_training_size = *json.get<int>("data_storage_trigger_training_size");
		transaction_count_per_model_update = *json.get<int>("transaction_count_per_model_update");
		enable_profiler = *json.get<bool>("enable_profiler");
		ml_model_stream_compressed_filter_limit = *json.get<float>("ml_model_stream_compressed_filter_limit");
		ml_model_stream_type = *json.get<std::string>("ml_model_stream_type");
		
		data_injector_inject_amount = *json.get<int>("data_injector_inject_amount");
		data_injector_inject_interval_scale_ms_to_tick = *json.get<int>("data_injector_inject_interval_scale_ms_to_tick");
	}
	
private:
	std::string generate_dataset_mode_type() const
	{
		switch (dataset_mode)
		{
			case dataset_mode_type::unknown:
				return "unknown";
				break;
			case dataset_mode_type::default_dataset:
				return "default";
				break;
			case dataset_mode_type::iid_dataset:
				return "iid";
				break;
			case dataset_mode_type::non_iid_dataset:
				return "non-iid";
				break;
		}
		throw std::logic_error("unreachable");
	}
};

using model_datatype = float;

int main(int argc, char **argv)
{
	//get current time
	const time_t current_time = time_util::get_current_utc_time();
	
	//register node types
	register_node_types<model_datatype>();
	
    //log file path
    google::InitGoogleLogging(argv[0]);
    std::filesystem::path log_path(DEPLOY_LOG_PATH);
    if (!std::filesystem::exists(log_path)) std::filesystem::create_directories(log_path);
    google::SetLogDestination(google::INFO, log_path.c_str());
    google::SetStderrLogging(google::INFO);

    //load configuration
    configuration_file deployment_config;
	configuration_file simulation_config;
	{
		deployment_config.SetDefaultConfiguration(get_default_DFL_deployment_configuration());
		{
			auto return_code = deployment_config.LoadConfiguration("./large_scale_DFL_deployment_config.json");
			if(return_code < configuration_file::NoError)
			{
				if (return_code == configuration_file::FileFormatError)
					LOG(FATAL) << "configuration file format error";
				else
					LOG(FATAL) << "configuration file error code: " << return_code;
			}
			//check correctness of deployment_config file (file exists?)
			auto [status, msg] = check_config(deployment_config);
			if (!status)
			{
				LOG(FATAL) << msg;
				return -1;
			}
		}
		
		std::string simulation_config_path;
		{
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
				std::cout <<  "First, you need to have a simulation configuration file, this tool will convert the simulation deployment_config to a DFL deployment directory." << std::endl;
				std::cout <<  "1) ./large_scale_DFL_deployment --> load generate deployment_config file to ../simulation/simulator_config.json" << std::endl;
				std::cout <<  "2) ./large_scale_DFL_deployment {path} --> load generate deployment_config file to certain path" << std::endl;
				return -1;
			}
		}
		
		std::filesystem::path config_file_path(simulation_config_path);
		{
			if (!std::filesystem::exists(config_file_path))
			{
				LOG(FATAL) << simulation_config_path << " does not exist.";
				return -1;
			}
			auto return_code = simulation_config.LoadConfiguration(simulation_config_path);
			if(return_code < configuration_file::NoError)
			{
				if (return_code == configuration_file::FileFormatError)
					LOG(FATAL) << "simulation configuration file format error";
				else
					LOG(FATAL) << "simulation configuration file error code: " << return_code;
			}
		}
	}
 
	//read configuration to node_deploy_info_container.
	std::map<std::string, node_deploy_info> node_deploy_info_container;
	
	auto simulation_config_json = simulation_config.get_json();
	
	std::vector<int> ml_dataset_all_possible_labels = *simulation_config.get_vec<int>("ml_dataset_all_possible_labels");
	std::vector<float> ml_non_iid_normal_weight = *simulation_config.get_vec<float>("ml_non_iid_normal_weight");
	LOG_IF(FATAL, ml_non_iid_normal_weight.size() != 2) << "ml_non_iid_normal_weight must be a two-value array, {max min}";
	
	//apply simulation node information to node deployment
	{
		auto nodes_json = simulation_config_json["nodes"];
		for (auto& single_node : nodes_json)
		{
			const std::string node_name = single_node["name"];
			
			node_deploy_info temp;
			temp.set_default_values();
			temp.name = node_name;
			temp.buffer_size = single_node["buffer_size"];
			
			//inject interval
			std::vector<int> interval_ticks = single_node["training_interval_tick"].get<std::vector<int>>();
			temp.data_injector_inject_interval_tick = std::accumulate(interval_ticks.begin(), interval_ticks.end(), 0.0f) / interval_ticks.size();
			
			//dataset mode
			{
				const std::string dataset_mode_str = single_node["dataset_mode"];
				if (dataset_mode_str == "default")
				{
					temp.dataset_mode = dataset_mode_type::default_dataset;
				}
				else if (dataset_mode_str == "iid")
				{
					temp.dataset_mode = dataset_mode_type::iid_dataset;
				}
				else if (dataset_mode_str == "non-iid")
				{
					temp.dataset_mode = dataset_mode_type::non_iid_dataset;
				}
				else
				{
					LOG(FATAL) << "unknown dataset_mode:" << dataset_mode_str;
					return -1;
				}
			}
			
			//label_distribution
			std::string dataset_mode = single_node["dataset_mode"];
			configuration_file::json non_iid_distribution = single_node["non_iid_distribution"];
			for (auto non_iid_item = non_iid_distribution.begin(); non_iid_item != non_iid_distribution.end(); ++non_iid_item)
			{
				int label = std::stoi(non_iid_item.key());
				auto min_max_array = *non_iid_item;
				float min = min_max_array.at(0);
				float max = min_max_array.at(1);
				if (max > min)
				{
					temp.special_non_iid_distribution[label] = {min, max};
				}
				else
				{
					temp.special_non_iid_distribution[label] = {max, min}; //swap the order
				}
			}
			for (auto &el: ml_dataset_all_possible_labels)
			{
				auto iter_el = temp.special_non_iid_distribution.find(el);
				if (iter_el == temp.special_non_iid_distribution.end())
				{
					//not set before
					temp.special_non_iid_distribution[el] = {ml_non_iid_normal_weight[0], ml_non_iid_normal_weight[1]};
				}
			}
			
			//node type
			const std::string node_type_str = single_node["node_type"];
			{
				auto result = node<model_datatype>::get_node_type_by_str(node_type_str);
				if (result)
				{
					temp.node_malicious_type = *result;
				}
				else
				{
					LOG(FATAL) << "unknown node type:" << node_type_str;
				}
			}
			
			node_deploy_info_container.emplace(node_name, std::move(temp));
		}
	}
	
	//update network topology
	auto node_topology_json = simulation_config_json["node_topology"];
	for (auto& topology_item : node_topology_json)
	{
		const std::string topology_item_str = topology_item.get<std::string>();
		auto average_degree_loc = topology_item_str.find(simulation_config_format::average_degree);
		auto unidirectional_loc = topology_item_str.find(simulation_config_format::unidirectional_term);
		auto bilateral_loc = topology_item_str.find(simulation_config_format::bilateral_term);
		
		if (topology_item_str == simulation_config_format::fully_connect)
		{
			LOG(INFO) << "network topology is fully connect";
			
			for (auto&[node_name, node_inst] : node_deploy_info_container)
			{
				for (auto&[target_node_name, target_node_inst] : node_deploy_info_container)
				{
					if (node_name != target_node_name)
					{
						node_inst.preferred_peers.push_back(target_node_inst.blockchain_address);
					}
				}
			}
			break;
		}
		else if (average_degree_loc != std::string::npos)
		{
			std::string degree_str = topology_item_str.substr(average_degree_loc + simulation_config_format::average_degree.length());
			int degree = std::stoi(degree_str);
			LOG(INFO) << "network topology is average degree: " << degree;
			LOG_IF(FATAL, degree > node_deploy_info_container.size() - 1) << "degree > node_count - 1, impossible to reach such large degree";
			
			std::vector<std::string> node_name_list;
			node_name_list.reserve(node_deploy_info_container.size());
			for (const auto&[node_name, node_inst] : node_deploy_info_container)
			{
				node_name_list.push_back(node_name);
			}
			
			std::random_device dev;
			std::mt19937 rng(dev());
			LOG(INFO) << "network topology average degree process begins";
			for (auto&[target_node_name, target_node_inst] : node_deploy_info_container)
			{
				std::shuffle(std::begin(node_name_list), std::end(node_name_list), rng);
				for (const std::string &connect_node_name : node_name_list)
				{
					//check average degree first to ensure there are already some connects.
					if (target_node_inst.preferred_peers.size() >= degree) break;
					if (connect_node_name == target_node_name) continue; // connect_node == self
					if (*std::find(target_node_inst.preferred_peers.begin(), target_node_inst.preferred_peers.end(), connect_node_name) == connect_node_name) continue;// connection already exists
					auto &connect_node = node_deploy_info_container.at(connect_node_name);
					target_node_inst.preferred_peers.push_back(connect_node.blockchain_address);
					LOG(INFO) << "network topology average degree process: " << target_node_name << " -> " << connect_node_name << "\t" << target_node_inst.blockchain_address.substr(0,8) << "->" << connect_node.blockchain_address.substr(0,8);
				}
			}
			LOG(INFO) << "network topology average degree process ends";
		}
		else if (unidirectional_loc != std::string::npos)
		{
			std::string lhs_node_str = topology_item_str.substr(0, unidirectional_loc);
			std::string rhs_node_str = topology_item_str.substr(unidirectional_loc + simulation_config_format::unidirectional_term.length());
			LOG(INFO) << "network topology: unidirectional connect " << lhs_node_str << " to " << rhs_node_str;
			auto lhs_node_iter = node_deploy_info_container.find(lhs_node_str);
			auto rhs_node_iter = node_deploy_info_container.find(rhs_node_str);
			LOG_IF(FATAL, lhs_node_iter == node_deploy_info_container.end()) << lhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
			LOG_IF(FATAL, rhs_node_iter == node_deploy_info_container.end()) << rhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
			
			if (*std::find(lhs_node_iter->second.preferred_peers.begin(), lhs_node_iter->second.preferred_peers.end(), rhs_node_str) == rhs_node_str)// connection already exists
			{
				LOG(WARNING) << rhs_node_str << " is already a peer of " << lhs_node_str;
			}
			else
			{
				auto &connect_node = node_deploy_info_container.at(rhs_node_str);
				lhs_node_iter->second.preferred_peers.push_back(connect_node.blockchain_address);
			}
		}
		else if (bilateral_loc != std::string::npos)
		{
			std::string lhs_node_str = topology_item_str.substr(0, bilateral_loc);
			std::string rhs_node_str = topology_item_str.substr(bilateral_loc + simulation_config_format::unidirectional_term.length());
			LOG(INFO) << "network topology: bilateral connect " << lhs_node_str << " with " << rhs_node_str;
			auto lhs_node_iter = node_deploy_info_container.find(lhs_node_str);
			auto rhs_node_iter = node_deploy_info_container.find(rhs_node_str);
			LOG_IF(FATAL, lhs_node_iter == node_deploy_info_container.end()) << lhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
			LOG_IF(FATAL, rhs_node_iter == node_deploy_info_container.end()) << rhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
			
			if (*std::find(lhs_node_iter->second.preferred_peers.begin(), lhs_node_iter->second.preferred_peers.end(), rhs_node_str) == rhs_node_str)// connection already exists
			{
				LOG(WARNING) << rhs_node_str << " is already a peer of " << lhs_node_str;
			}
			else
			{
				auto &rhs_connected_node = node_deploy_info_container.at(rhs_node_str);
				lhs_node_iter->second.preferred_peers.push_back(rhs_connected_node.blockchain_address);
			}
			
			if (*std::find(rhs_node_iter->second.preferred_peers.begin(), rhs_node_iter->second.preferred_peers.end(), lhs_node_str) == lhs_node_str)// connection already exists
			{
				LOG(WARNING) << lhs_node_str << " is already a peer of " << rhs_node_str;
			}
			else
			{
				auto &lhs_connected_node = node_deploy_info_container.at(lhs_node_str);
				rhs_node_iter->second.preferred_peers.push_back(lhs_connected_node.blockchain_address);
			}
		}
		else
		{
			LOG(FATAL) << "unknown topology item: " << topology_item_str;
		}
	}
	
	//apply information to node deployment_config.
	for (auto& [node_name, node_target] : node_deploy_info_container)
	{
		node_target.apply_deployment_information(deployment_config);
	}
	
	//allocate port
	{
		uint16_t port_start = *deployment_config.get<uint16_t>("port_start");
		uint16_t port_end = *deployment_config.get<uint16_t>("port_end");
		std::vector<uint16_t> available_ports;
		available_ports.reserve(port_end-port_start);
		for (auto i = port_start; i < port_end; ++i) available_ports.push_back(i);
		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(available_ports.begin(), available_ports.end(), g);
		size_t count = 0;
		for (auto& [node_name, node_target] : node_deploy_info_container)
		{
			node_target.data_storage_service_port = available_ports[count];
			count++;
			LOG_IF(FATAL, count == available_ports.size()) << "not enough ports here";
		}
		for (auto& [node_name, node_target] : node_deploy_info_container)
		{
			node_target.port = available_ports[count];
			count++;
			LOG_IF(FATAL, count == available_ports.size()) << "not enough ports here";
		}
		
	}
	
	//generate node deployment
	std::filesystem::path dfl_executable_path(*deployment_config.get<std::string>("path_exe_DFL"));
	std::filesystem::path dfl_injector_path(*deployment_config.get<std::string>("path_exe_injector"));
	std::filesystem::path dfl_reputation_file_path(*deployment_config.get<std::string>("path_dll_reputation"));
	std::filesystem::path dfl_introducer_path(*deployment_config.get<std::string>("path_exe_introducer"));
	
	std::filesystem::path output_path = std::string("deployment_") + time_util::time_to_text(current_time);
	if (!std::filesystem::exists(output_path)) std::filesystem::create_directories(output_path);
	
	//generate DFL introducer
	LOG(INFO) << "generating introducer node";
	std::vector<introducer_node> introducer_nodes;
	{
		std::string ip = *deployment_config.get<std::string>("introducer_ip");
		uint16_t port = *deployment_config.get<uint16_t>("introducer_port");
		introducer_nodes.emplace_back(ip, port);
	}
	for (const auto& single_introducer_node : introducer_nodes)
	{
		std::filesystem::path current_node_output_path = output_path / ("introducer-" + single_introducer_node.blockchain_address.substr(0, 8));
		if (!std::filesystem::exists(current_node_output_path)) std::filesystem::create_directories(current_node_output_path);
		
		std::filesystem::copy_file(dfl_introducer_path, current_node_output_path / dfl_introducer_path.filename());
		configuration_file dfl_introducer_config_file;
		dfl_introducer_config_file.SetDefaultConfiguration(single_introducer_node.generate_json_for_introducer());
		auto status = dfl_introducer_config_file.LoadConfiguration(current_node_output_path/CONFIG_FILE_NAME::DFL_INTRODUCER);
		LOG_IF(FATAL, status!= configuration_file::FileNotFoundAndGenerateOne) << "logic error";
	}
	
	
	//put DFL introducer info in DFL nodes.
	for (auto& [node_name, node_target] : node_deploy_info_container)
	{
		node_target.introducer_nodes = introducer_nodes;
	}
	
	//generate DFL nodes
	for (const auto& [node_name, node_target] : node_deploy_info_container)
	{
		LOG(INFO) << "generating node: " << node_name;
		
		std::filesystem::path current_node_output_path = output_path / node_name;
		if (!std::filesystem::exists(current_node_output_path)) std::filesystem::create_directories(current_node_output_path);
		
		//copy DFL executable files
		std::filesystem::copy_file(dfl_executable_path, current_node_output_path / dfl_executable_path.filename());
		std::filesystem::copy_file(dfl_injector_path, current_node_output_path / dfl_injector_path.filename());
		std::filesystem::copy_file(dfl_reputation_file_path, current_node_output_path / dfl_reputation_file_path.filename());
		
		//copy dataset
		{
			std::filesystem::path temp(node_target.dataset_label_path);
			std::filesystem::copy_file(temp, current_node_output_path / temp.filename());
		}
		{
			std::filesystem::path temp(node_target.dataset_path);
			std::filesystem::copy_file(temp, current_node_output_path / temp.filename());
		}
		
		std::ofstream lenet_solver(current_node_output_path/compile_time_content::lenet_solver_memory_name);
		lenet_solver << compile_time_content::lenet_solver_memory_content;
		lenet_solver.close();
		std::ofstream lenet_model(current_node_output_path/compile_time_content::lenet_train_memory_name);
		lenet_model << compile_time_content::lenet_train_memory;
		lenet_model.close();
		
		{
			configuration_file dfl_config_file;
			dfl_config_file.SetDefaultConfiguration(node_target.generate_dfl_node_config());
			auto status = dfl_config_file.LoadConfiguration(current_node_output_path/CONFIG_FILE_NAME::DFL_EXE);
			LOG_IF(FATAL, status!= configuration_file::FileNotFoundAndGenerateOne) << "logic error";
		}
		{
			configuration_file dfl_data_injector_config_file;
			dfl_data_injector_config_file.SetDefaultConfiguration(node_target.generate_dfl_node_data_injector_config());
			auto status = dfl_data_injector_config_file.LoadConfiguration(current_node_output_path/CONFIG_FILE_NAME::DFL_DATA_INJECTOR);
			LOG_IF(FATAL, status!= configuration_file::FileNotFoundAndGenerateOne) << "logic error";
		}
		
		
	}
	
	
	
    return 0;
}