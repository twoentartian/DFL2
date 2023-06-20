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

std::optional<std::vector<std::tuple<int, int>>> generate_network_topology(std::map<int, int> peer_count_of_each_node, bool bilateral = true)
{
    auto node_count = peer_count_of_each_node.size();
    static std::random_device rd;
    std::mt19937_64 rng(rd());
    
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
        node_available_nodes[node].erase(node);//remove self from available nodes.
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
            std::advance(it, dist(rng));
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
        
        // remove current node in from all node_available_nodes
        if (bilateral)
        {
            for (int node = 0; node < node_count; ++node)
            {
                node_available_nodes[node].erase(node_name);
            }
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

std::optional<std::vector<std::tuple<int, int>>> generate_network_topology_no_map(std::map<int, int> peer_count_of_each_node, bool bilateral = true)
{
    auto node_count = peer_count_of_each_node.size();
    static std::random_device rd;
    std::mt19937_64 rng(rd());
    
    std::vector<std::tuple<int, int>> connections;
    
    //set free peers
    std::set<int> free_nodes;
    {
        for (const auto& [node, _] : peer_count_of_each_node)
        {
            free_nodes.insert(node);
        }
    }
    
    //set available peer count
    auto* available_peer_count = new int[node_count];
    {
        for (int i = 0; i < node_count; ++i)
        {
            available_peer_count[i] = 0;
        }
        for (const auto& [node, peer_count] : peer_count_of_each_node)
        {
            available_peer_count[node] = peer_count;
        }
    }
    
    //set available nodes
    auto* node_available_nodes = new std::set<size_t>[node_count];
    {
        std::set<size_t> all_nodes;
        for (size_t node = 0; node < node_count; ++node)
        {
            all_nodes.emplace(node);
        }
        for (size_t node = 0; node < node_count; ++node)
        {
            node_available_nodes[node] = all_nodes;
            node_available_nodes[node].erase(node);//remove self from available nodes.
        }
    }
    
    auto find_maximum_available_peer = [](int* available_peer_count, size_t node_count) -> size_t {
        int maximum_value = available_peer_count[0];
        size_t maximum_index = 0;
        
        for (size_t i = 0; i < node_count; ++i)
        {
            if (available_peer_count[i] > maximum_value)
            {
                maximum_value = available_peer_count[i];
                maximum_index = i;
            }
        }
        return maximum_index;
    };
    
    bool success = true;
    while (true)
    {
        size_t process_index = find_maximum_available_peer(available_peer_count, node_count);
        size_t available_peers = available_peer_count[process_index];
        if (available_peers == 0) // we are done!
        {
            break;
        }
        
        size_t try_count = 0;
        while (available_peer_count[process_index] != 0)
        {
            try_count++;
            if (try_count == 10000)
            {
                success = false;
                break;
            }
            
            std::uniform_int_distribution<size_t> dist(0, free_nodes.size()-1);
            auto it = std::begin(free_nodes);
            std::advance(it, dist(rng));
            auto random_generated_peer = *it;
            if (available_peer_count[random_generated_peer] == 0) { //that peer is already fully connected
                free_nodes.erase(random_generated_peer);
                continue;
            }
            
            if (process_index == random_generated_peer) continue;
            if (!node_available_nodes[process_index].contains(random_generated_peer)) continue; //already a peer
            
            //we decide to connect to this node
            if (bilateral)
            {
                available_peer_count[process_index]--;
                node_available_nodes[process_index].erase(random_generated_peer);
                available_peer_count[random_generated_peer]--;
                node_available_nodes[random_generated_peer].erase(process_index);
            }
            else
            {
                available_peer_count[process_index]--;
            }
            
            connections.emplace_back(process_index, random_generated_peer);
        }
        
        if (!success) break;
    }
    
    delete[] available_peer_count;
    delete[] node_available_nodes;
    
    if (success)
    {
        return {connections};
    }
    else
    {
        return {};
    }
}

void apply_generator_config_to_output_config(const configuration_file::json& generator_json, configuration_file::json& output_json, const std::string& comment, bool delete_other_comments = false)
{
    if (delete_other_comments)
    {
        std::vector<std::string> erase_list;
        for (auto &json_item: output_json.items())
        {
            if (json_item.key().starts_with("comment"))
                erase_list.push_back(json_item.key());
        }
        for (const auto &erase_item_name: erase_list)
        {
            output_json.erase(erase_item_name);
        }
    }
    
    output_json[comment] = generator_json;
}