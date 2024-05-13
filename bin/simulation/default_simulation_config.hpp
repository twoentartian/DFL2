#pragma once

#include <tuple>
#include <configure_file.hpp>
#include <os_info.hpp>
#include "global_types.hpp"

configuration_file::json get_default_simulation_configuration()
{
	configuration_file::json output;

    //simulator_opti_specific
    output["simulator_opti_averaging_algorithm"] = "train_average_standard";
    output["simulator_opti_averaging_algorithm_all_possible"] = {"train_average_standard", "train_50_average_50", "train_50_average_50_fix_variance_auto", "train_100_average_0", "train_0_average_100", "train_0_average_100_fix_variance_auto"};
    configuration_file::json opti_averaging_algorithm_args;
    opti_averaging_algorithm_args["beta"] = "0.5";
    opti_averaging_algorithm_args["variance_correction"] = "true";
    opti_averaging_algorithm_args["variance_correction_method"] = "self";
    opti_averaging_algorithm_args["variance_correction_method_all_possible"] = "self,others,follow_beta";
    opti_averaging_algorithm_args["skip_layers"] = "";
    opti_averaging_algorithm_args["skip_layers_example"] = "conv1,conv2,conv3,ip1";
    output["simulator_opti_averaging_algorithm_args"] = opti_averaging_algorithm_args;

	output["report_time_remaining_per_tick_elapsed"] = 100;

    output["random_training_sequence"] = true;

    output["ml_dataset_type"] = "mnist";
    output["ml_model_type_candidate"] = "mnist,cifar10";
	output["ml_solver_proto"] = "../../../dataset/MNIST/lenet_solver_memory.prototxt";
	output["ml_train_dataset"] = "../../../dataset/MNIST/train-images.idx3-ubyte";
	output["ml_train_dataset_label"] = "../../../dataset/MNIST/train-labels.idx1-ubyte";
	output["ml_test_dataset"] = "../../../dataset/MNIST/t10k-images.idx3-ubyte";
	output["ml_test_dataset_label"] = "../../../dataset/MNIST/t10k-labels.idx1-ubyte";
	
	output["ml_max_tick"] = 10000;
	output["ml_train_batch_size"] = 64;
	output["ml_test_batch_size"] = 100;
	output["ml_non_iid_normal_weight"] = configuration_file::json::array({10.0, 15.0});
	output["ml_dataset_all_possible_labels"] = configuration_file::json::array({0,1,2,3,4,5,6,7,8,9});

    output["early_stop_enable"] = false;
    output["early_stop_threshold_accuracy"] = 0.9;
    output["early_stop_threshold_node_ratio"] = 0.9;
	
	if (os_info::get_os_type() == os_info::os_type::linux_series)
	{
		output["ml_reputation_dll_path"] = "../reputation_sdk/sample/libreputation_50_training_50_averaging.so";
	}
	else if (os_info::get_os_type() == os_info::os_type::apple)
	{
		output["ml_reputation_dll_path"] = "../reputation_sdk/sample/libreputation_50_training_50_averaging.dylib";
	}
	else
	{
		LOG(FATAL) << "OS type not supported";
	}
	
	configuration_file::json node;
	configuration_file::json node_non_iid = configuration_file::json::object();
	node_non_iid["1"] = configuration_file::json::array({1.0,2.0});
	node_non_iid["3"] = configuration_file::json::array({2.0,3.0});
	node["name"] = "1";
	node["dataset_mode"] = "default"; //default - randomly choose from dataset, iid - randomly choose from iid labels, non-iid - choose higher frequency labels for specific label
	node["training_interval_tick"] = configuration_file::json::array({8,9,10,11,12});
	node["buffer_size"] = 2;
	node["model_generation_type"] = "compressed"; //normal, compressed
	node["filter_limit"] = 0.5;
    node["first_train_tick"] = 0;
	node["node_type"] = "normal";
    node["node_type_arg"] = "none";
	node["non_iid_distribution"] = node_non_iid;
	
	configuration_file::json nodes = configuration_file::json::array();
	nodes.push_back(node);
	node["name"] = "2";
	node["dataset_mode"] = "iid";
	node["node_type"] = "observer";
	nodes.push_back(node);
	
	output["nodes"] = nodes;
	
	output["node_topology"] = configuration_file::json::array({"fully_connect", "average_degree-2", "1->2", "1--2"});
	
	configuration_file::json services = configuration_file::json::object();
	{
		configuration_file::json accuracy_service = configuration_file::json::object();
		accuracy_service["enable"] = true;
		accuracy_service["interval"] = 20;
        accuracy_service["fixed_test_dataset"] = true;
        accuracy_service["ignore_disabled_node"] = false;
		services["accuracy"] = accuracy_service;
	}
	{
		configuration_file::json weights_service = configuration_file::json::object();
		weights_service["enable"] = true;
		weights_service["interval"] = 20;
		services["model_weights_difference_record"] = weights_service;
	}
    {
        configuration_file::json model_abs_change_during_averaging = configuration_file::json::object();
        model_abs_change_during_averaging["enable"] = false;
        services["model_abs_change_during_averaging"] = model_abs_change_during_averaging;
    }
    {
        configuration_file::json model_weights_variance_record_service = configuration_file::json::object();
        model_weights_variance_record_service["enable"] = true;
        model_weights_variance_record_service["interval"] = 20;
        services["model_weights_variance_record"] = model_weights_variance_record_service;
    }
	{
		configuration_file::json force_broadcast_service = configuration_file::json::object();
		force_broadcast_service["enable"] = false;
        force_broadcast_service["sample_config_items"] = {"after_train:at:10,20,30,40",
                                                          "after_train:every:10",
                                                          "before_train:every:10,",
                                                          "before_average:at:0",
                                                          "after_average:every:100",
                                                          "before_average:at:0:{node_name}"};
		force_broadcast_service["config_items"] = {"before_train:at:0", "after_train:every:100"};
		services["force_broadcast_average"] = force_broadcast_service;
	}
	{
		configuration_file::json peer_control_service = configuration_file::json::object();
		peer_control_service["enable"] = false;
		peer_control_service["least_peer_change_interval"] = 50;
		peer_control_service["fedavg_buffer_size"] = "linear"; //// candidates: static, linear
		peer_control_service["accuracy_threshold_high"] = 0.8;
		peer_control_service["accuracy_threshold_low"] = 0.2;
		services["time_based_hierarchy_service"] = peer_control_service;
	}
    {
        configuration_file::json reputation_record_service = configuration_file::json::object();
        reputation_record_service["enable"] = false;
        services["reputation_record"] = reputation_record_service;
    }
    {
        configuration_file::json delta_weight_after_training_averaging_record_service = configuration_file::json::object();
        delta_weight_after_training_averaging_record_service["enable"] = false;
        delta_weight_after_training_averaging_record_service["path"] = "delta_weight";
        delta_weight_after_training_averaging_record_service["nodes_to_record"] = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19";
        delta_weight_after_training_averaging_record_service["layers_to_record"] = "all,conv1";
        services["delta_weight_after_training_averaging_record"] = delta_weight_after_training_averaging_record_service;
    }
    {
        configuration_file::json apply_delta_weight = configuration_file::json::object();
        apply_delta_weight["enable"] = false;
        apply_delta_weight["config"] = "0:train, 0:init, 0:average";
        services["apply_delta_weight"] = apply_delta_weight;
    }
    {
        configuration_file::json model_record_service = configuration_file::json::object();
        model_record_service["enable"] = false;
        model_record_service["path"] = "./models";
        model_record_service["interval"] = 1000;
        model_record_service["nodes"] = "0,1,2,3,4";
        model_record_service["final_record"] = true;
        services["model_record"] = model_record_service;
    }
    {
        configuration_file::json received_modeL_record = configuration_file::json::object();
        received_modeL_record["enable"] = false;
        received_modeL_record["path"] = "./received_models";
        received_modeL_record["nodes"] = "0,1,2,3,4";
        services["received_model_record"] = received_modeL_record;
    }
    {
        configuration_file::json stage_manager = configuration_file::json::object();
        stage_manager["enable"] = false;
        stage_manager["script_path"] = "./script.json";
        services["stage_manager"] = stage_manager;
    }
    {
        configuration_file::json apply_received_model = configuration_file::json::object();
        apply_received_model["enable"] = false;
        apply_received_model["path"] = "./received_models";
        services["apply_received_model"] = apply_received_model;
    }
    {
        configuration_file::json compiled_services = configuration_file::json::object();
        compiled_services["enable_model_randomness"] = false;
        compiled_services["init_model_randomness"] = 0.5;
        services["compiled_services"] = compiled_services;
    }

    {
        configuration_file::json network_topology_manager_service = configuration_file::json::object();

        configuration_file::json network_topology_manager_service_read_from_file = configuration_file::json::object();
        network_topology_manager_service_read_from_file["enable"] = false;
        network_topology_manager_service_read_from_file["topology_file_path"] = "./topology.json";
        network_topology_manager_service["read_from_file"] = network_topology_manager_service_read_from_file;

        configuration_file::json network_topology_manager_service_rebuild_scale_free_network = configuration_file::json::object();
        network_topology_manager_service_rebuild_scale_free_network["enable"] = false;
        network_topology_manager_service_rebuild_scale_free_network["gamma"] = 3.0;
        network_topology_manager_service_rebuild_scale_free_network["min_peer"] = 3;
        network_topology_manager_service_rebuild_scale_free_network["buffer_to_peer_ratio"] = 1.0;
        network_topology_manager_service_rebuild_scale_free_network["interval"] = 200;
        network_topology_manager_service["scale_free_network"] = network_topology_manager_service_rebuild_scale_free_network;

        configuration_file::json network_topology_manager_service_connection_pair_swap = configuration_file::json::object();
        network_topology_manager_service_connection_pair_swap["enable"] = false;
        network_topology_manager_service_connection_pair_swap["percentage"] = 0.5;
        network_topology_manager_service_connection_pair_swap["interval"] = 200;
        network_topology_manager_service["connection_pair_swap"] = network_topology_manager_service_connection_pair_swap;

        services["network_topology_manager"] = network_topology_manager_service;
    }
	output["services"] = services;
	
	return output;
}
