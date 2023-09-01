#include <iostream>
#include <filesystem>
#include <random>
#include <regex>

#include <glog/logging.h>

#include <configure_file.hpp>

int main(int argc, char* argv[]) {
    configuration_file config;
    std::string config_file;
    int target_node = 0;
    if (argc == 3) {
        config_file.assign(argv[1]);
        target_node = std::stoi(argv[2]);
    } else if (argc == 2) {
        config_file.assign("../simulation/simulator_config.json");
        target_node = std::stoi(argv[1]);
    } else {
        std::cout << "topology_remove_receive [config_file_path] [node_name]" << std::endl;
        std::cout << "topology_remove_receive [node_name]" << std::endl;
        return -1;
    }
    std::cout << "removing receive edge of node: " << target_node << std::endl;

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

    configuration_file::json& topology_json = config.get_json()["node_topology"];
    CHECK(topology_json.is_array());

    std::regex r(R"((\d+)(\D+)(\d+))");

    configuration_file::json new_topology_json = topology_json;
    new_topology_json.clear();
    for (auto& edge: topology_json) {
        std::vector<int> nodes;
        std::string dir;
        {
            std::string edge_str = edge;
            std::smatch match;
            if (std::regex_search(edge_str, match, r)) {
                nodes.push_back(std::stoi(match[1]));
                nodes.push_back(std::stoi(match[3]));
                dir = match[2];
            }
        }
        assert(nodes.size() == 2);
        if (nodes[0] == target_node) {
            std::stringstream ss;
            if (dir == "--") {
                ss << nodes[0] << "->" << nodes[1];
            }
            else if (dir == "->") {
                ss << nodes[0] << "->" << nodes[1];
            }
            else {
                std::cout << "unknown edge type: " << dir << std::endl;
                return -1;
            }
            new_topology_json.push_back(ss.str());
        }
        else if (nodes[1] == target_node) {
            std::stringstream ss;
            if (dir == "--") {
                ss << nodes[1] << "->" << nodes[0];
            }
            else if (dir == "->") {
                //remove this edge
            }
            else {
                std::cout << "unknown edge type: " << dir << std::endl;
                return -1;
            }
            new_topology_json.push_back(ss.str());
        }
        else /*target node doesn't appear, so we just continue*/ {
            new_topology_json.push_back(edge);
        }
    }

    config.get_json()["node_topology"] = new_topology_json;

    std::cout << "writing back" << std::endl;
    config.write_back();
    return 0;
}