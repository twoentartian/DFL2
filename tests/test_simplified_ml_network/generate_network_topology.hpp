#pragma once

void add_to_mainland(int mainland_node, const std::map<int, std::set<int>> &peer_map, std::set<int> &mainland)
{
    if (mainland.contains(mainland_node)) return;
    mainland.emplace(mainland_node);
    for (auto &peer: peer_map.at(mainland_node))
    {
        add_to_mainland(peer, peer_map, mainland);
    }
};

std::optional<std::tuple<std::vector<std::tuple<int, int>>, size_t>> generate_network_topology(size_t number_of_nodes, size_t number_of_peers)
{
    bool flop_connection = (number_of_peers > number_of_nodes / 2); //if true, peer = all nodes - peers
    int node_peer_connection_count_override = flop_connection ? (number_of_nodes - 1 - number_of_peers) : number_of_peers;
    if (number_of_peers * number_of_nodes % 2 != 0)
        return {};
    
    size_t try_count = 0;
    bool whole_success = false;
    std::vector<std::tuple<int, int>> connections;
    
    while (try_count < 10000)
    {
        try_count++;
        bool success = true;
        static std::random_device rd;
        static std::mt19937 g(rd());
        
        //init variables
        connections.clear();
        
        std::map<int, int> node_instances_counter;
        std::map<int, std::set<int>> node_ban_list;
        std::map<int, std::set<int>> node_available_nodes;
        std::set<int> all_nodes;
        for (int node = 0; node < number_of_nodes; ++node)
        {
            all_nodes.emplace(node);
        }
        for (int node = 0; node < number_of_nodes; ++node)
        {
            node_instances_counter[node] = node_peer_connection_count_override;
            node_available_nodes[node] = all_nodes;
            for (int node_to_remove = 0; node_to_remove < node; ++node_to_remove)
            {
                node_available_nodes[node].erase(node_to_remove);
            }
        }
        
        //try to generate network
        for (auto &[node_name, instance]: node_instances_counter)
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
                std::advance(it, dist(g));
                auto random_pick_node = *it;
                node_available_nodes[node_name].erase(it);//remove this picked node
                if (random_pick_node == node_name) continue;
                if (node_ban_list[node_name].contains(random_pick_node))
                    continue;
                if (node_instances_counter[random_pick_node] == 0) continue;
                
                //ban both nodes
                node_ban_list[node_name].emplace(random_pick_node);
                node_ban_list[random_pick_node].emplace(node_name);
                
                node_instances_counter[node_name]--;
                node_instances_counter[random_pick_node]--;
                connections.emplace_back(node_name, random_pick_node);
            }
        }
        if (success)
        {
//check whether we get a network with islands
            std::map<int, std::set<int>> peer_map;
            for (auto &[node, peer]: connections)
            {
                peer_map[node].emplace(peer);
                peer_map[peer].emplace(node);
            }
            if (flop_connection)
            {
                std::set<int> whole_set;
                for (auto &[node, all_peers]: peer_map)
                {
                    whole_set.emplace(node);
                }
                for (auto &[node, all_peers]: peer_map)
                {
                    std::set<int> flopped_connections;
                    std::set_difference(whole_set.begin(), whole_set.end(), all_peers.begin(), all_peers.end(), std::insert_iterator<std::set<int>>(flopped_connections, flopped_connections.begin()));
                    peer_map[node] = flopped_connections;
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
    
    if (flop_connection)
    {
        //we need to flop back
        std::map<int, std::set<int>> real_connections;
        std::set<int> all_nodes;
        for (int node = 0; node < number_of_nodes; ++node)
        {
            all_nodes.emplace(node);
        }
        for (int node = 0; node < number_of_nodes; ++node)
        {
            real_connections[node] = all_nodes;
            real_connections[node].erase(node);
        }
        for (auto &[node0, node1]: connections)
        {
            real_connections[node0].erase(node1);
        }
        
        connections.clear();
        
        for (auto &[node, peers]: real_connections)
        {
            for (auto &peer: peers)
            {
                connections.push_back({node, peer});
            }
        }
    }
    
    return {{connections, try_count}};
}
