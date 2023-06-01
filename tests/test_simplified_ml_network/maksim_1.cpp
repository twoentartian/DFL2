#include <optional>
#include <random>
#include <numeric>

#define BOOST_TEST_MAIN

#include <boost/test/included/unit_test.hpp>

#include <util.hpp>
#include "generate_network_topology.hpp"

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
    float real_model;
    std::set<simplified_ml_network_node *> peers;
    int node_name;
    size_t next_training_tick;
    
};

BOOST_AUTO_TEST_SUITE (simplified_ml_network)
    
    BOOST_AUTO_TEST_CASE (maksim_1)
    {
        size_t number_of_nodes = 400;
//        float accuracy_increase_per_training = 1.0 / (20000.0 / 10);
        float accuracy_increase_per_training = 0.0;
        float accuracy_increase_per_training_stddev = 0.0;
        std::vector<size_t> training_tick = {8, 9, 10, 11, 12};
        size_t number_of_peers = 8;
        size_t model_buffer_size = 8;
        size_t simulation_tick = 1000;
        size_t record_accuracy_per_tick = 1;
        
        float conservative = 0.5;
        
        //init
        static std::random_device rd{};
        std::mt19937 gen{rd()};
        
        //set node topology
        auto result = generate_network_topology(number_of_nodes, number_of_peers);
        BOOST_CHECK(result);
        auto [topology, try_count] = *result;
        
        //generate network
        std::map<int, simplified_ml_network_node> all_nodes;
        {
            std::uniform_real_distribution<float> dist(0, 0.01);
            for (int i = 0; i < number_of_nodes; ++i)
            {
                auto iter = all_nodes.emplace(i, simplified_ml_network_node(i));
                iter.first->second.model = dist(gen);
            }
            for (auto &[a, b]: topology)
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
        
        //generate real values
        std::uniform_real_distribution<float> dis{0.0, 1.0};
        for (auto &[node_name, node]: all_nodes)
        {
            node.real_model = dis(gen);
        }
        
        for (int tick = 0; tick < simulation_tick; ++tick)
        {
            std::cout << "tick: " << tick << std::endl;
            for (auto &[name_name, node]: all_nodes)
            {
                if (node.next_training_tick == tick)
                {
                    node.next_training_tick += *util::select_randomly(training_tick.begin(), training_tick.end());
                    
                    node.model = (node.model + node.real_model) / 2;
                    
                    //broadcast
                    for (auto &peer: node.peers)
                    {
                        peer->model_buffer.push_back(node.model);
                        if (peer->model_buffer.size() == model_buffer_size)
                        {
                            //averaging
                            auto model_buffer_average = std::reduce(peer->model_buffer.begin(), peer->model_buffer.end()) / peer->model_buffer.size();
                            peer->model = conservative * peer->model + (1 - conservative) * model_buffer_average;
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