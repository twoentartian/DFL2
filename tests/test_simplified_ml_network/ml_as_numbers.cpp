#include <optional>
#include <random>
#include <numeric>

#define BOOST_TEST_MAIN

#include <boost/test/included/unit_test.hpp>

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
    if(number_of_peers * number_of_nodes % 2 != 0)
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
                if (node_ban_list[node_name].contains(random_pick_node)) continue;
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

class simplified_ml_network_node
{
public:
    simplified_ml_network_node(int name)
    {
        model = 0;
        node_name = name;
        next_training_tick = 0;
    }
    
    std::vector<float> model_buffer;
    float model;
    std::set<simplified_ml_network_node*> peers;
    int node_name;
    size_t next_training_tick;
    
};

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator &g)
{
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

BOOST_AUTO_TEST_SUITE (simplified_ml_network)

    BOOST_AUTO_TEST_CASE (ml_as_numbers)
    {
        size_t number_of_nodes = 100;
//        float accuracy_increase_per_training = 1.0 / (20000.0 / 10);
        float accuracy_increase_per_training = 0.0;
        float accuracy_increase_per_training_stddev = 0.0;
        std::vector<size_t> training_tick = {8, 9, 10, 11, 12};
        size_t number_of_peers = 8;
        size_t model_buffer_size = 8;
        size_t simulation_tick = 200;
        size_t record_accuracy_per_tick = 1;
        
        float conservative = 0.5;

        //init
        std::random_device rd{};
        std::mt19937 gen{rd()};

        //set node topology
        auto result = generate_network_topology(number_of_nodes, number_of_peers);
        BOOST_CHECK(result);
        auto [topology, try_count] = *result;
        
        //generate network
        std::map<int, simplified_ml_network_node> all_nodes;
        {
            std::uniform_real_distribution<float> dist(0,1);
            for (int i = 0; i < number_of_nodes; ++i)
            {
                auto iter = all_nodes.emplace(i, simplified_ml_network_node(i));
                iter.first->second.model = dist(gen);
            }
            for (auto& [a, b]: topology)
            {
                auto node_a = all_nodes.find(a);
                auto node_b = all_nodes.find(b);
                node_a->second.peers.emplace(&node_b->second);
                node_b->second.peers.emplace(&node_a->second);
            }
        }

        //begin simulation
        std::ofstream accuracy_file("./accuracy.csv");
        accuracy_file << "tick";
        for (auto &[name_name, _]: all_nodes)
        {
            accuracy_file << "," << name_name;
        }
        accuracy_file << std::endl;

        std::normal_distribution<float> dist{accuracy_increase_per_training, accuracy_increase_per_training_stddev};
        
        for (int tick = 0; tick < simulation_tick; ++tick)
        {
            std::cout << "tick: " << tick << std::endl;
            for (auto& [name_name, node]: all_nodes)
            {
                if (node.next_training_tick == tick)
                {
                    node.next_training_tick += *select_randomly(training_tick.begin(), training_tick.end());

                    node.model += dist(gen);
                    
                    //broadcast
                    for (auto& peer: node.peers)
                    {
                        peer->model_buffer.push_back(node.model);
                        if (peer->model_buffer.size() == model_buffer_size)
                        {
                            //averaging
                            auto model_buffer_average = std::reduce(peer->model_buffer.begin(), peer->model_buffer.end()) / peer->model_buffer.size();
                            peer->model = conservative * peer->model + (1-conservative) * model_buffer_average;
                            peer->model_buffer.clear();
                        }
                    }
                }
            }
    
            if (tick % record_accuracy_per_tick == 0)
            {
                accuracy_file << tick;
                for (auto &[_, node]: all_nodes)
                {
                    accuracy_file << "," << node.model;
                }
                accuracy_file << std::endl;
            }
        }
    
        accuracy_file.flush();
        accuracy_file.close();
    }

BOOST_AUTO_TEST_SUITE_END()