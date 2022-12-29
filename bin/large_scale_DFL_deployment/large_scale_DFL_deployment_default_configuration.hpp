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
	
	output["path_mnist_dataset_label"] = "../../../dataset/MNIST/train-labels.idx1-ubyte";
	output["path_mnist_dataset_data"] = "../../../dataset/MNIST/train-images.idx3-ubyte";
	output["data_injector_inject_amount"] = 8;
	output["data_injector_inject_interval_scale_ms_to_tick"] = 100;
    output["data_injector_inject_interval_variance"] = 500;
	
	output["port_start"] = 10000;
	output["port_end"] = 60000;
	
	output["reputation_dll_datatype"] = "float";
	output["timeout_second"] = 0;
	output["data_storage_service_concurrency"] = 2;
	output["ml_test_batch_size"] = 100;
	
	output["ml_model_stream_type"] = "normal";
	output["ml_model_stream_compressed_filter_limit"] = 0.5;
	
	output["blockchain_estimated_block_size"] = 10;
	output["data_storage_trigger_training_size"] = 64;
	output["transaction_count_per_model_update"] = 10;
	output["enable_profiler"] = false;
	output["network_inactive_peer_second"] = 60;
	output["network_use_preferred_peers_only"] = false;
	output["network_maximum_peer"] = 10;
	
	output["introducer_port"] = 8500;
	output["introducer_ip"] = "127.0.0.1";
	
    return output;
}

std::tuple<bool, std::string> check_config(const configuration_file& config_json)
{
    const std::vector<std::string> path_check = {"path_exe_DFL", "path_exe_injector", "path_exe_introducer", "path_dll_reputation", "path_mnist_dataset_label", "path_mnist_dataset_data"};

    for (auto&& item: path_check) {
        auto exe_path_opt = config_json.get<std::string>(item);
        if (!exe_path_opt) return {false, "configuration {" + item + "} is empty"};
        std::filesystem::path exe_path(*exe_path_opt);
        if (!std::filesystem::exists(exe_path))
        {
            return {false, "configuration {" + item + "}, path{" + *exe_path_opt + "} doesn't exist"};
        }
    }
    
    return {true, ""};
}
