//
// Created by tyd, 25-Feb-23.
//

#pragma once

#include <set>
#include <map>
#include <random>

#include <util.hpp>
#include <configure_file.hpp>

void add_to_mainland(int mainland_node, const std::map<int, std::set<int>> &peer_map, std::set<int> &mainland)
{
    if (mainland.contains(mainland_node)) return;
    mainland.emplace(mainland_node);
    for (auto &peer: peer_map.at(mainland_node))
    {
        add_to_mainland(peer, peer_map, mainland);
    }
};

template<typename RANDOM_ENGINE = std::mt19937>
std::optional<std::vector<std::tuple<int, int>>> generate_network_topology(int node_count, std::map<int, int> peer_count_of_each_node, bool bilateral = true, RANDOM_ENGINE random_engine = std::mt19937(std::random_device()()))
{
    std::vector<std::tuple<int, int>> connections;
    
    std::map<int, int>& node_instances_counter = peer_count_of_each_node;
    std::map<int, std::set<int>> node_ban_list;
    std::map<int, std::set<int>> node_available_nodes;
    std::set<int> all_nodes;
    for (int node = 0; node < node_count; ++node)
    {
        all_nodes.emplace(node);
    }
    for (int node = 0; node < node_count; ++node)
    {
        node_available_nodes[node] = all_nodes;
        
        if (bilateral)  //nodes shouldn't connect to a smaller node, e.g. 10 cannot connect to 9.
        {
            for (int node_to_remove = 0; node_to_remove <= node; ++node_to_remove)
            {
                node_available_nodes[node].erase(node_to_remove);
            }
        }
    }
    
    //find the node with more connections
    auto nodes_with_more_connections = util::sort_map_according_to_value(node_instances_counter, true);
    
    //try to generate network
    bool success = true;
    for (auto &[node_name, instance]: nodes_with_more_connections)
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
            
            std::uniform_int_distribution dist(0, int(node_available_nodes[node_name].size()) - 1);
            auto it = std::begin(node_available_nodes[node_name]);
            std::advance(it, dist(random_engine));
            auto random_pick_node = *it;
            node_available_nodes[node_name].erase(it);//remove this picked node
            if (random_pick_node == node_name) continue;
            if (node_ban_list[node_name].contains(random_pick_node)) continue;
            if (node_instances_counter[random_pick_node] == 0) continue;
            
            if (bilateral)
            {
                node_ban_list[node_name].emplace(random_pick_node);
                node_ban_list[random_pick_node].emplace(node_name);
                node_instances_counter[node_name]--;
                node_instances_counter[random_pick_node]--;
            }
            else
            {
                node_ban_list[node_name].emplace(random_pick_node);
                node_instances_counter[node_name]--;
            }

            connections.emplace_back(node_name, random_pick_node);
        }
    }
    
    if (success)
    {
        return {connections};
    }
    else
    {
        return {};
    }
}

void apply_generator_config_to_output_config(const configuration_file::json& generator_json, configuration_file::json& output_json, const std::string& comment)
{
    output_json[comment] = generator_json;
}