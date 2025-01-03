#include <thread>
#include <unordered_map>
#include <fstream>
#include <set>
#include <atomic>
#include <chrono>
#include <csignal>
#include <execinfo.h>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/format.hpp>

#include <glog/logging.h>

#include <configure_file.hpp>
#include <crypto.hpp>
#include <util.hpp>
#include <time_util.hpp>
#include <ml_layer.hpp>
#include <thread_pool.hpp>
#include <dll_importer.hpp>
#include <utility>
#include <memory_consumption.hpp>

#include "../reputation_sdk.hpp"
#include "./default_simulation_config.hpp"
#include "./node.hpp"
#include "./simulation_service.hpp"
#include "./simulation_config_format.hpp"
#include "./simulator_opti_model_update.hpp"

/** assumptions in this simulator:
 *  (1) no transaction transmission time
 *  (2) no blockchain overhead
 *
 */

using model_datatype = float;

std::map<std::string, node<model_datatype> *> node_container;
std::unordered_map<std::string, std::shared_ptr<opti_model_update<model_datatype>>> node_model_update;
////set model updating algorithm
//using model_updating_algorithm = train_50_average_50<model_datatype>;
//using model_updating_algorithm = train_50_average_50_fix_variance_auto<model_datatype>;
std::string model_updating_algorithm_name;

void handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

int main(int argc, char *argv[])
{
    signal(SIGSEGV, handler);

	constexpr char config_file_path[] = "./simulator_config.json";

    //register model averaging algorithm
    register_model_updating_algorithms<model_datatype>();

	//register node types
	register_node_types<model_datatype>();
	
	//create new folder
	std::string time_str = time_util::time_to_text(time_util::get_current_utc_time());
	std::filesystem::path output_path = std::filesystem::current_path() / time_str;
	std::filesystem::create_directories(output_path);
	
	//log file path
	google::InitGoogleLogging(argv[0]);
	std::filesystem::path log_path(output_path / "log");
	if (!std::filesystem::exists(log_path)) std::filesystem::create_directories(log_path);
	google::SetLogDestination(google::INFO, (log_path.string() + "/").c_str());
	
	//load configuration
	configuration_file config;
	config.SetDefaultConfiguration(get_default_simulation_configuration());
    auto load_config_rc = config.LoadConfiguration(config_file_path, {
            "/simulator_opti_averaging_algorithm_args"_json_pointer,
            "/services"_json_pointer, "/services/accuracy"_json_pointer, "/services/apply_delta_weight"_json_pointer,
            "/services/apply_received_model"_json_pointer,"/services/compiled_services"_json_pointer,
            "/services/delta_weight_after_training_averaging_record"_json_pointer,
            "/services/force_broadcast_average"_json_pointer,"/services/model_abs_change_during_averaging"_json_pointer,
            "/services/model_record"_json_pointer,
            "/services/model_weights_difference_record"_json_pointer,"/services/model_weights_variance_record"_json_pointer,
            "/services/network_topology_manager"_json_pointer,
            "/services/network_topology_manager/connection_pair_swap"_json_pointer,"/services/network_topology_manager/read_from_file"_json_pointer,"/services/network_topology_manager/scale_free_network"_json_pointer,
            "/services/received_model_record"_json_pointer,"/services/reputation_record"_json_pointer,"/services/stage_manager"_json_pointer,
            "/services/time_based_hierarchy_service"_json_pointer}); //"services/accuracy" is due to the "fixed_test_dataset" field
    if (load_config_rc < 0)
	{
		LOG(FATAL) << "cannot load configuration file, wrong format?";
		return -1;
	}
	auto config_json = config.get_json();
	//backup configuration file
	std::filesystem::copy(config_file_path, output_path / "simulator_config.json");
	
	//update global var
    model_updating_algorithm_name = *config.get<std::string>("simulator_opti_averaging_algorithm");
    std::map<std::string, std::string> model_updating_algorithm_args;
    {
        if (config_json.contains("simulator_opti_averaging_algorithm_args")) {
            const auto& simulator_opti_averaging_algorithm_args_json = config_json["simulator_opti_averaging_algorithm_args"];
            LOG_ASSERT(simulator_opti_averaging_algorithm_args_json.is_object());
            for (const auto& it : simulator_opti_averaging_algorithm_args_json.items()) {
                model_updating_algorithm_args[it.key()] = it.value();
            }
        }
    }
    auto ml_dataset_type = *config.get<std::string>("ml_dataset_type");
    auto random_training_sequence = *config.get<bool>("random_training_sequence");
	auto ml_solver_proto = *config.get<std::string>("ml_solver_proto");
	auto ml_train_dataset = *config.get<std::string>("ml_train_dataset");
	auto ml_train_dataset_label = *config.get<std::string>("ml_train_dataset_label");
	auto ml_test_dataset = *config.get<std::string>("ml_test_dataset");
	auto ml_test_dataset_label = *config.get<std::string>("ml_test_dataset_label");
	
	auto ml_max_tick = *config.get<int>("ml_max_tick");
	auto ml_train_batch_size = *config.get<int>("ml_train_batch_size");
	auto ml_test_batch_size = *config.get<int>("ml_test_batch_size");
	
	auto early_stop_enable = *config.get<bool>("early_stop_enable");
	auto early_stop_threshold_accuracy = *config.get<float>("early_stop_threshold_accuracy");
	auto early_stop_threshold_node_ratio = *config.get<float>("early_stop_threshold_node_ratio");
	
	auto report_time_remaining_per_tick_elapsed = *config.get<int>("report_time_remaining_per_tick_elapsed");
	
	std::vector<int> ml_dataset_all_possible_labels = *config.get_vec<int>("ml_dataset_all_possible_labels");
	std::vector<float> ml_non_iid_normal_weight = *config.get_vec<float>("ml_non_iid_normal_weight");
	LOG_IF(ERROR, ml_non_iid_normal_weight.size() != 2) << "ml_non_iid_normal_weight must be a two-value array, {max min}";
	auto ml_reputation_dll_path = config.get<std::string>("ml_reputation_dll_path");
	LOG_IF(WARNING, ml_reputation_dll_path.has_value()) << "ml_reputation_dll will be ignored in this simulator_opti because it is compiled in the simulator.";

#pragma region load node configurations
	auto nodes_json = config_json["nodes"];
	for (auto &single_node: nodes_json) {
        const std::string node_name = single_node["name"];
        {
            auto iter = node_container.find(node_name);
            if (iter != node_container.end()) {
                LOG(FATAL) << "duplicate node name";
                return -1;
            }
        }

        //name
        const int buf_size = single_node["buffer_size"];

        const std::string node_type = single_node["node_type"];
        std::optional<std::string> node_type_arg = {};
        if (single_node.contains("node_type_arg")) {
            node_type_arg = std::make_optional(single_node["node_type_arg"].get<std::string>());
        }

        node<model_datatype> *temp_node = nullptr;

        //find the node in the registered node map
        {
            auto result = node<model_datatype>::get_node_by_type(node_type);
            if (result == nullptr) {
                LOG(FATAL) << "unknown node type:" << node_type;
            } else {
                temp_node = result->new_node(node_name, buf_size, node_type_arg);
            }
        }

        auto [iter, status] = node_container.emplace(node_name, temp_node);
        //add to model update buffer
        auto model_buffer = opti_model_update<model_datatype>::create_update_algorithm_from_name(model_updating_algorithm_name);
        LOG_IF(FATAL, !model_buffer.has_value()) << model_updating_algorithm_name << " is not a valid updating algorithm";
        node_model_update.emplace(node_name, *model_buffer);

        //load models solver
        iter->second->solver->load_caffe_model(ml_solver_proto);

        //dataset mode
        const std::string dataset_mode_str = single_node["dataset_mode"];
        if (dataset_mode_str == "default") {
            iter->second->dataset_mode = dataset_mode_type::default_dataset;
        } else if (dataset_mode_str == "iid") {
            iter->second->dataset_mode = dataset_mode_type::iid_dataset;
        } else if (dataset_mode_str == "non-iid") {
            iter->second->dataset_mode = dataset_mode_type::non_iid_dataset;
        } else {
            LOG(FATAL) << "unknown dataset_mode:" << dataset_mode_str;
            return -1;
        }

        //model_generation_type
        const std::string model_generation_type_str = single_node["model_generation_type"];
        if (model_generation_type_str == "normal") {
            iter->second->model_generation_type = Ml::model_compress_type::normal;
        } else if (model_generation_type_str == "compressed_by_diff" || model_generation_type_str == "random_sampling") {
            LOG(FATAL) << "{model_generation_type == compressed} will be ignored in this simulator_opti because the simulator_opti only supports normal.";
            iter->second->model_generation_type = Ml::model_compress_type::compressed_by_diff;
        } else {
            LOG(FATAL) << "unknown model_generation_type:" << model_generation_type_str;
            return -1;
        }

        //filter_limit
        iter->second->filter_limit = single_node["filter_limit"];

        //first_train_tick
        if (single_node.contains("first_train_tick")) {
            iter->second->next_train_tick = single_node["first_train_tick"];
        }

        //label_distribution
        std::string dataset_mode = single_node["dataset_mode"];
        if (dataset_mode == "iid") {
            //nothing to do because the single_node["non_iid_distribution"] will not be used
        } else if (dataset_mode == "non-iid") {
            configuration_file::json non_iid_distribution = single_node["non_iid_distribution"];
            for (auto non_iid_item = non_iid_distribution.begin();
                 non_iid_item != non_iid_distribution.end(); ++non_iid_item) {
                int label = std::stoi(non_iid_item.key());
                auto min_max_array = *non_iid_item;
                float min = min_max_array.at(0);
                float max = min_max_array.at(1);
                if (max > min) {
                    iter->second->special_non_iid_distribution[label] = {min, max};
                } else {
                    iter->second->special_non_iid_distribution[label] = {max, min}; //swap the order
                }
            }
            for (auto &el: ml_dataset_all_possible_labels) {
                auto iter_el = iter->second->special_non_iid_distribution.find(el);
                if (iter_el == iter->second->special_non_iid_distribution.end()) {
                    //not set before
                    iter->second->special_non_iid_distribution[el] = {ml_non_iid_normal_weight[0],
                                                                      ml_non_iid_normal_weight[1]};
                }
            }
        } else if (dataset_mode == "default") {
            //nothing to do because the single_node["non_iid_distribution"] will not be used
        } else {
            LOG(ERROR) << "unknown dataset_mode:" << single_node["dataset_mode"];
        }

        //training_interval_tick
        for (auto &el: single_node["training_interval_tick"]) {
            iter->second->training_interval_tick.push_back(el);
        }
    }
#pragma endregion

#pragma region load network topology configuration
	/** network topology configuration
	 * you can use fully_connect, average_degree-{degree}, 1->2, 1--2, the topology items' order in the configuration file determines the order of adding connections.
	 * fully_connect: connect all nodes, and ignore all other topology items.
	 * average_degree-: connect the network to reach the degree for all nodes. If there are previous added topology, average_degree will add connections
	 * 					until reaching the degree and no duplicate connections.
	 * 1->2: add 2 as the peer of 1.
	 * 1--2: add 2 as the peer of 1 and 1 as the peer of 2.
	 */
	{
		auto node_topology_json = config_json["node_topology"];
		for (auto &topology_item : node_topology_json)
		{
			const std::string topology_item_str = topology_item.get<std::string>();
			auto average_degree_loc = topology_item_str.find(simulation_config_format::average_degree);
			auto unidirectional_loc = topology_item_str.find(simulation_config_format::unidirectional_term);
			auto bilateral_loc = topology_item_str.find(simulation_config_format::bilateral_term);
			
			auto check_duplicate_peer = [](const node<model_datatype> &target_node, const std::string &peer_name) -> bool
			{
				if (target_node.planned_peers.find(peer_name) == target_node.planned_peers.end())
					return false;
				else
					return true;
			};
			
			if (topology_item_str == simulation_config_format::fully_connect)
			{
				LOG(INFO) << "network topology is fully connect";
				
				for (auto&[node_name, node_inst] : node_container)
				{
					for (auto&[target_node_name, target_node_inst] : node_container)
					{
						if (node_name != target_node_name)
						{
							node_inst->planned_peers.emplace(target_node_inst->name, target_node_inst);
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
				LOG_IF(FATAL, degree > node_container.size() - 1) << "degree > node_count - 1, impossible to reach such large degree";
				
				std::vector<std::string> node_name_list;
				node_name_list.reserve(node_container.size());
				for (const auto&[node_name, node_inst] : node_container)
				{
					node_name_list.push_back(node_name);
				}
				
				static std::random_device dev;
				static std::mt19937 rng(dev());
				LOG(INFO) << "network topology average degree process begins";
				for (auto&[target_node_name, target_node_inst] : node_container)
				{
					std::shuffle(std::begin(node_name_list), std::end(node_name_list), rng);
					for (const std::string &connect_node_name : node_name_list)
					{
						//check average degree first to ensure there are already some connects.
						if (target_node_inst->planned_peers.size() >= degree) break;
						if (connect_node_name == target_node_name) continue; // connect_node == self
						if (check_duplicate_peer((*target_node_inst), connect_node_name)) continue; // connection already exists
						
						auto &connect_node = node_container.at(connect_node_name);
						target_node_inst->planned_peers.emplace(connect_node->name, connect_node);
						LOG(INFO) << "network topology average degree process: " << target_node_inst->name << " -> " << connect_node->name;
					}
				}
				LOG(INFO) << "network topology average degree process ends";
			}
			else if (unidirectional_loc != std::string::npos)
			{
				std::string lhs_node_str = topology_item_str.substr(0, unidirectional_loc);
				std::string rhs_node_str = topology_item_str.substr(unidirectional_loc + simulation_config_format::unidirectional_term.length());
				LOG(INFO) << "network topology: unidirectional connect " << lhs_node_str << " to " << rhs_node_str;
				auto lhs_node_iter = node_container.find(lhs_node_str);
				auto rhs_node_iter = node_container.find(rhs_node_str);
				LOG_IF(FATAL, lhs_node_iter == node_container.end()) << lhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
				LOG_IF(FATAL, rhs_node_iter == node_container.end()) << rhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
				
				if (check_duplicate_peer(*(lhs_node_iter->second), rhs_node_str))// connection already exists
				{
					LOG(WARNING) << rhs_node_str << " is already a peer of " << lhs_node_str;
				}
				else
				{
					auto &connect_node = node_container.at(rhs_node_str);
					lhs_node_iter->second->planned_peers.emplace(connect_node->name, connect_node);
				}
			}
			else if (bilateral_loc != std::string::npos)
			{
				std::string lhs_node_str = topology_item_str.substr(0, bilateral_loc);
				std::string rhs_node_str = topology_item_str.substr(bilateral_loc + simulation_config_format::unidirectional_term.length());
				LOG(INFO) << "network topology: bilateral connect " << lhs_node_str << " with " << rhs_node_str;
				auto lhs_node_iter = node_container.find(lhs_node_str);
				auto rhs_node_iter = node_container.find(rhs_node_str);
				LOG_IF(FATAL, lhs_node_iter == node_container.end()) << lhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
				LOG_IF(FATAL, rhs_node_iter == node_container.end()) << rhs_node_str << " is not found in nodes, raw topology: " << topology_item_str;
				
				if (check_duplicate_peer(*(lhs_node_iter->second), rhs_node_str))// connection already exists
				{
					LOG(WARNING) << rhs_node_str << " is already a peer of " << lhs_node_str;
				}
				else
				{
					auto &rhs_connected_node = node_container.at(rhs_node_str);
					lhs_node_iter->second->planned_peers.emplace(rhs_connected_node->name, rhs_connected_node);
				}
				
				if (check_duplicate_peer(*(rhs_node_iter->second), lhs_node_str))// connection already exists
				{
					LOG(WARNING) << lhs_node_str << " is already a peer of " << rhs_node_str;
				}
				else
				{
					auto &lhs_connected_node = node_container.at(lhs_node_str);
					rhs_node_iter->second->planned_peers.emplace(lhs_connected_node->name, lhs_connected_node);
				}
			}
			else
			{
				LOG(ERROR) << "unknown topology item: " << topology_item_str;
			}
		}
	}
#pragma endregion
	
	//load dataset
	Ml::data_converter<model_datatype> train_dataset;
    Ml::data_converter<model_datatype> test_dataset;
    if (ml_dataset_type == "mnist") {
        train_dataset.load_dataset_mnist(ml_train_dataset, ml_train_dataset_label);
        test_dataset.load_dataset_mnist(ml_test_dataset, ml_test_dataset_label);
    }
    else if (ml_dataset_type == "cifar10") {
        train_dataset.load_dataset_cifar10(ml_train_dataset, Ml::load_dataset_type::TRAIN);
        test_dataset.load_dataset_cifar10(ml_test_dataset, Ml::load_dataset_type::TEST);
    }
    else {
        LOG(FATAL) << "unknown ml_model_type:" << ml_dataset_type;
    }
	
	//node vector container
	std::vector<node<model_datatype>*> node_pointer_vector_container;
	node_pointer_vector_container.reserve(node_container.size());
	for (auto &single_node : node_container)
	{
		node_pointer_vector_container.push_back(single_node.second);
	}
	
	////services
	std::unordered_map<std::string, std::shared_ptr<service<model_datatype>>> services;
    //trigger service
    const auto trigger_service = [&services](int tick, service_trigger_type trigger_type){
        for (auto& [name, service_instance]: services)
        {
            service_instance->process_per_tick(tick, trigger_type);
        }
    };

	{
		services.emplace("accuracy", new accuracy_record<model_datatype>());
		services.emplace("model_weights_difference_record", new model_weights_difference_record<model_datatype>());
        services.emplace("model_weights_variance_record", new model_weights_variance_record<model_datatype>());
		services.emplace("force_broadcast_average", new force_broadcast_model<model_datatype>());
        services.emplace("model_abs_change_during_averaging", new model_abs_change_during_averaging<model_datatype>());
		services.emplace("time_based_hierarchy_service", new time_based_hierarchy_service<model_datatype>());
		services.emplace("reputation_record", new reputation_record<model_datatype>());
		services.emplace("model_record", new model_record<model_datatype>());
		services.emplace("network_topology_manager", new network_topology_manager<model_datatype>());
        services.emplace("delta_weight_after_training_averaging_record", new delta_weight_after_training_averaging_record<model_datatype>());
        services.emplace("apply_delta_weight", new apply_delta_weight<model_datatype>());
        services.emplace("received_model_record", new received_model_record<model_datatype>());
        services.emplace("apply_received_model", new apply_received_model<model_datatype>());
        services.emplace("variance_control", new variance_control<model_datatype>());
        services.emplace("stage_manager", new stage_manager_service<model_datatype>());
        services.emplace("compiled_services", new compiled_services<model_datatype>());
		auto services_json = config_json["services"];
		LOG_IF(FATAL, services_json.is_null()) << "services are not defined in configuration file";

		{
			auto check_and_get_config = [&services_json](const std::string& service_name) -> auto{
				auto json_config = services_json[service_name];
				LOG_IF(FATAL, json_config.is_null()) << "service: \"" << service_name << "\" config item is empty";
				return json_config;
			};
			
			//accuracy service
			{
				auto service_iter = services.find("accuracy");
				
				std::static_pointer_cast<accuracy_record<model_datatype>>(service_iter->second)->ml_solver_proto = ml_solver_proto;
				std::static_pointer_cast<accuracy_record<model_datatype>>(service_iter->second)->test_dataset = &test_dataset;
				std::static_pointer_cast<accuracy_record<model_datatype>>(service_iter->second)->ml_test_batch_size = ml_test_batch_size;
				
				service_iter->second->apply_config(check_and_get_config("accuracy"));
				service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
			}
			
			//model weights difference record
			{
				auto service_iter = services.find("model_weights_difference_record");
				
				service_iter->second->apply_config(check_and_get_config("model_weights_difference_record"));
				service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
			}
            
            //model weights variance record
            {
                auto service_iter = services.find("model_weights_variance_record");
                
                service_iter->second->apply_config(check_and_get_config("model_weights_variance_record"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }
			
			//force_broadcast
			{
				auto service_iter = services.find("force_broadcast_average");
				
				service_iter->second->apply_config(check_and_get_config("force_broadcast_average"));
				service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
			}

            //model_abs_change_during_averaging
            {
                auto service_iter = services.find("model_abs_change_during_averaging");

                service_iter->second->apply_config(check_and_get_config("model_abs_change_during_averaging"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }
			
			//time_based_hierarchy_service
			{
				auto service_iter = services.find("time_based_hierarchy_service");
				
				std::static_pointer_cast<time_based_hierarchy_service<model_datatype>>(service_iter->second)->ml_solver_proto = ml_solver_proto;
				std::static_pointer_cast<time_based_hierarchy_service<model_datatype>>(service_iter->second)->test_dataset = &test_dataset;
				std::static_pointer_cast<time_based_hierarchy_service<model_datatype>>(service_iter->second)->ml_test_batch_size = ml_test_batch_size;
				std::static_pointer_cast<time_based_hierarchy_service<model_datatype>>(service_iter->second)->ml_dataset_all_possible_labels = &ml_dataset_all_possible_labels;
				
				service_iter->second->apply_config(check_and_get_config("time_based_hierarchy_service"));
				service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
			}
			
			//reputation_record
			{
				auto service_iter = services.find("reputation_record");
				
				service_iter->second->apply_config(check_and_get_config("reputation_record"));
				service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
			}
			
			//model record
			{
				auto service_iter = services.find("model_record");
				std::static_pointer_cast<model_record<model_datatype>>(service_iter->second)->total_tick = ml_max_tick;
				
				service_iter->second->apply_config(check_and_get_config("model_record"));
				service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
			}

            //received_model_record
            {
                auto service_iter = services.find("received_model_record");

                service_iter->second->apply_config(check_and_get_config("received_model_record"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }

            //apply_received_model
            {
                auto service_iter = services.find("apply_received_model");
                auto add_model_to_update_buffer = [](const std::string& node_name, const Ml::caffe_parameter_net<model_datatype>& model){
                    node_model_update[node_name]->add_model(model);
                };
                std::static_pointer_cast<apply_received_model<model_datatype>>(service_iter->second)->send_model_to_update_buffer_handler = add_model_to_update_buffer;

                service_iter->second->apply_config(check_and_get_config("apply_received_model"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }
			
			//network_topology_manager
			{
				auto service_iter = services.find("network_topology_manager");
				
				service_iter->second->apply_config(check_and_get_config("network_topology_manager"));
				service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
			}

            //delta_weight_after_training_averaging_record
            {
                auto service_iter = services.find("delta_weight_after_training_averaging_record");

                service_iter->second->apply_config(check_and_get_config("delta_weight_after_training_averaging_record"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }

            //apply_delta_weight
            {
                auto service_iter = services.find("apply_delta_weight");

                service_iter->second->apply_config(check_and_get_config("apply_delta_weight"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }
            
            //variance_control
            {
                auto service_iter = services.find("variance_control");
    
                service_iter->second->apply_config(check_and_get_config("variance_control"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }

            //stage_manager_service
            {
                auto service_iter = services.find("stage_manager");

                service_iter->second->apply_config(check_and_get_config("stage_manager"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }

            //compiled_services
            {
                auto service_iter = services.find("compiled_services");

                service_iter->second->apply_config(check_and_get_config("compiled_services"));
                service_iter->second->init_service(output_path, node_container, node_pointer_vector_container);
            }

			//final service check
			{
				LOG_IF(FATAL, services["network_topology_manager"]->enable && services["time_based_hierarchy_service"]->enable) << "you cannot enable time_based_hierarchy_service and network_topology_manager at same time";
			}
		}
	}
    //prepare "process_on_event" services
    const auto& received_model_record_service = services["received_model_record"];
    const auto& model_abs_change_during_averaging_service = services["model_abs_change_during_averaging"];
	const auto& variance_control_service = services["variance_control"];
 
	////////////  BEGIN SIMULATION  ////////////
	std::mutex accuracy_container_lock;
	std::map<std::string, float> accuracy_container;
	for (const auto& [node_name, node]: node_container) {
        node->node_init();
		accuracy_container[node_name] = 0.0;
	}
	
	{
		auto last_time_point = std::chrono::system_clock::now();
		
		int tick = 0;
		while (tick <= ml_max_tick)
		{
			std::cout << "tick: " << tick << " (" << ml_max_tick << ")" << std::endl;
			LOG(INFO) << "tick: " << tick << " (" << ml_max_tick << ")";

            //services
            trigger_service(tick, service_trigger_type::start_of_tick);

            ////report simulation speed
			if (tick != 0 && tick % report_time_remaining_per_tick_elapsed == 0)
			{
				auto now = std::chrono::system_clock::now();
				std::chrono::duration<float, std::milli> time_elapsed_ms = now - last_time_point;
				last_time_point = now;
				float speed_ms_per_tick = time_elapsed_ms.count() / float(report_time_remaining_per_tick_elapsed);
				std::chrono::milliseconds time_remain_ms(int(float(ml_max_tick - tick) * speed_ms_per_tick));
				std::time_t est_finish_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() + time_remain_ms);
				std::tm est_finish_time_tm = *std::localtime(&est_finish_time);
				std::cout << "speed: " << std::setprecision(2) << speed_ms_per_tick/1000 << "s/tick, est finish at: " << std::put_time( &est_finish_time_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
			}

            ////report memory consumption
            LOG(INFO) << "memory consumption (before training): " << get_memory_consumption_byte() / 1024 / 1024 << " MB";

            //services
            trigger_service(tick, service_trigger_type::start_of_training);

			////train the model
			tmt::ParallelExecution_StepIncremental([&tick, &train_dataset, &received_model_record_service, &model_abs_change_during_averaging_service, &variance_control_service, &ml_train_batch_size, &ml_dataset_all_possible_labels, random_training_sequence](uint32_t index, uint32_t thread_index, node<model_datatype>* single_node){
				if (tick >= single_node->next_train_tick)
				{
                    single_node->model_trained = true;
                    
                    std::vector<const Ml::tensor_blob_like<model_datatype>*> train_data, train_label;
                    std::tie(train_data, train_label) = get_dataset_by_node_type(train_dataset, *single_node, ml_train_batch_size, ml_dataset_all_possible_labels, random_training_sequence);
                    
                    static std::random_device dev;
                    static std::mt19937 rng(dev());
                    std::uniform_int_distribution<int> distribution(0, int(single_node->training_interval_tick.size()) - 1);
                    single_node->next_train_tick += single_node->training_interval_tick[distribution(rng)];
                    
                    {
                        auto parameter_before = single_node->solver->get_parameter();
                        single_node->simulation_service_data.model_before_training_ptr = &parameter_before;
                        
                        //model_abs_change_during_averaging service
                        model_abs_change_during_averaging_service->process_on_event(tick, service_trigger_type::model_before_training, single_node->name);
                        //variance control
                        variance_control_service->process_on_event(tick, service_trigger_type::model_before_training, single_node->name);
                    }

                    single_node->train_model(train_data, train_label, true);
                    auto output_opt = single_node->generate_model_sent();
                    if (!output_opt)
                    {
                        LOG(INFO) << "ignore output for node " << single_node->name << " at tick " << tick;
                    }
                    
                    {
                        auto parameter_after = single_node->solver->get_parameter();
                        single_node->simulation_service_data.model_after_training_ptr = &parameter_after;
                        
                        //model_abs_change_during_averaging service
                        model_abs_change_during_averaging_service->process_on_event(tick, service_trigger_type::model_after_training, single_node->name);
                        //variance_control
                        single_node->simulation_service_data.recent_training_data = train_data;
                        single_node->simulation_service_data.recent_training_label = train_label;
                        variance_control_service->process_on_event(tick, service_trigger_type::model_after_training, single_node->name);
                    }
                    
                    if (output_opt) {
                        //add ML network to FedAvg buffer
                        for (auto [updating_node_name, updating_node] : single_node->peers)
                        {
                            //only add send model to other nodes if they are enabled
                            if (!updating_node->enable) continue;
        
                            //allow peer node pre-processing the model
                            auto model_after_pre_processing = updating_node->preprocess_received_models(*output_opt);
                            node_model_update[updating_node_name]->add_model(model_after_pre_processing);
                            {
                                std::lock_guard guard(updating_node->simulation_service_data.just_received_model_ptr_lock);
                                updating_node->simulation_service_data.just_received_model_ptr = &model_after_pre_processing;
                                updating_node->simulation_service_data.just_received_model_source_node_name = single_node->name;
                                received_model_record_service->process_on_event(tick, service_trigger_type::model_added_to_average_buffer, updating_node_name);
                                model_abs_change_during_averaging_service->process_on_event(tick, service_trigger_type::model_added_to_average_buffer, updating_node_name);
                            }
                        }
                    }
				}
				else
				{
					single_node->model_trained = false;
				}
			}, node_pointer_vector_container.size(), node_pointer_vector_container.data());

            ////report memory consumption
            LOG(INFO) << "memory consumption (after training, before averaging): " << get_memory_consumption_byte() / 1024 / 1024 << " MB";

            //services
            trigger_service(tick, service_trigger_type::end_of_training);

            //services
            trigger_service(tick, service_trigger_type::start_of_averaging);

			////check fedavg buffer full
			tmt::ParallelExecution_StepIncremental([&model_updating_algorithm_args, &tick, &test_dataset, &model_abs_change_during_averaging_service, &variance_control_service, &ml_test_batch_size,&ml_dataset_all_possible_labels, &accuracy_container_lock, &accuracy_container](uint32_t index, uint32_t thread_index, node<model_datatype>* single_node){
				if (node_model_update[single_node->name]->get_model_count() >= single_node->buffer_size) {
					single_node->model_averaged = true;

					//update model
					auto parameter = single_node->solver->get_parameter();
                    
                    //trigger service
                    {
                        single_node->simulation_service_data.model_before_averaging_ptr = &parameter;
                        model_abs_change_during_averaging_service->process_on_event(tick, service_trigger_type::model_before_averaging, single_node->name);
                        variance_control_service->process_on_event(tick, service_trigger_type::model_before_averaging, single_node->name);
                    }
                    
					auto[test_data, test_label] = get_dataset_by_node_type(test_dataset, *single_node, ml_test_batch_size, ml_dataset_all_possible_labels);
					float self_accuracy = single_node->evaluate_model(test_data, test_label);

					single_node->last_measured_tick = tick;
					std::string log_msg = (boost::format("tick: %1%, node: %2%, accuracy: %3%") % tick % single_node->name % self_accuracy).str();
					printf("%s\n", log_msg.data());
					LOG(INFO) << log_msg;

                    parameter = node_model_update[single_node->name]->get_output_model(parameter, test_data, test_label, single_node->name, model_updating_algorithm_args);   //reset the model buffer and get the output
                    if (single_node->enable_averaging) {
                        single_node->pre_averaging_models();
                        single_node->solver->set_parameter(parameter);
                        single_node->post_averaging_models();   // allow node to post process the model after averaging
                    }
                    else {
                        LOG(INFO) << "skip averaging for node " << single_node->name << " at tick " << tick;
                    }

                    //trigger service
                    {
                        single_node->simulation_service_data.model_after_averaging_ptr = &parameter;
                        model_abs_change_during_averaging_service->process_on_event(tick, service_trigger_type::model_after_averaging, single_node->name);
                        variance_control_service->process_on_event(tick, service_trigger_type::model_after_averaging, single_node->name);
                    }

					//clear buffer and start new loop
					single_node->parameter_buffer.clear();

					//add self accuracy to accuracy container
					{
						std::lock_guard guard(accuracy_container_lock);
						accuracy_container[single_node->name] = self_accuracy;
					}
				}
				else
				{
					//do nothing
					single_node->model_averaged = false;
				}
			}, node_pointer_vector_container.size(), node_pointer_vector_container.data());

            //services
            trigger_service(tick, service_trigger_type::end_of_averaging);

            ////report memory consumption
            LOG(INFO) << "memory consumption (after averaging): " << get_memory_consumption_byte() / 1024 / 1024 << " MB";
            
			//services
            trigger_service(tick, service_trigger_type::end_of_tick);
			
			////early stop?
			if (early_stop_enable)
			{
				size_t counter_above_threshold = 0;
				for (const auto& [node_name, accuracy]: accuracy_container)
				{
					if (accuracy >= early_stop_threshold_accuracy)
					{
						counter_above_threshold++;
					}
				}
				auto node_ratio = static_cast<float>(counter_above_threshold) / static_cast<float>(accuracy_container.size());
				if (node_ratio > early_stop_threshold_node_ratio)
				{
					//early stop
					LOG(INFO) << "early stop, " << node_ratio << " of the nodes' accuracy is above " << early_stop_threshold_node_ratio;
					break;
				}
			}

            ////report memory consumption
            LOG(INFO) << "memory consumption (end of tick " << tick << "): " << get_memory_consumption_byte() / 1024 / 1024 << " MB";
            
			tick++;
		}
	}

    //destruction
	for (auto& [name, service_instance]: services)
	{
		service_instance->destruction_service();
	}
    for (auto& [_, ptr]: node_container)
    {
        delete ptr;
    }
    node<model_datatype>::deregister_all_node_types();
	
	return 0;
}
