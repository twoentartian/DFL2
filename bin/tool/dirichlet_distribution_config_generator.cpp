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
	double a = 0;
	if (argc == 3)
	{
		config_file.assign(argv[1]);
		a = std::stod(argv[2]);
	}
	else if (argc == 2)
	{
		config_file.assign("../simulation/simulator_config.json");
		a = std::stod(argv[1]);
	}
	else
	{
		std::cout << "dirichlet_distribution_config_generator [config_file_path] [alpha]" << std::endl;
		std::cout << "dirichlet_distribution_config_generator [alpha]" << std::endl;
		return -1;
	}
	std::cout << "alpha=" << a << std::endl;
	
	std::filesystem::path config_file_path(config_file);
	if (!std::filesystem::exists(config_file_path))
	{
		std::cerr << config_file << " does not exist" << std::endl;
		return -1;
	}
	
	config.LoadConfiguration(config_file);
	
	//all labels
	std::vector<int> all_labels = *config.get_vec<int>("ml_dataset_all_possible_labels");
    CHECK(!all_labels.empty());
	
	//nodes
	static std::random_device rd;
	std::gamma_distribution<double> gamma(a,10000/a);
	configuration_file::json& nodes_json = config.get_json()["nodes"];
 
	CHECK(nodes_json.is_array());
	
	for (auto& node : nodes_json)
	{
		if (node["node_type"] != "normal") continue;
		
		node["dataset_mode"] = "non-iid";
		auto& dis_json = node["non_iid_distribution"];
		for (auto single_label : all_labels)
		{
			auto value = static_cast<float>(gamma(rd));
			dis_json[std::to_string(single_label)] = {value,value};
			std::cout << "node-" << node["name"] << " label-" << single_label << " dis: " << value << std::endl;
		}
	}
    
    configuration_file::json generator_comment;
    generator_comment["alpha"] = a;
    apply_generator_config_to_output_config(generator_comment, config.get_json(), "comment_this_config_file_is_modified_by_distribution_config_generator", false);
    
	std::cout << "writing back" << std::endl;
	config.write_back();
	return 0;
}