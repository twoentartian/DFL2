#pragma once

#include <iostream>
#include <filesystem>
#include <map>
#include <mutex>

#include <glog/logging.h>
#include <boost/asio.hpp>
#include <boost_serialization_wrapper.hpp>

#include <ml_layer.hpp>
#include <measure_time.hpp>

#include "analyze_models.hpp"

std::map<int, std::map<std::string, float>> calculate_model_distance_from_starting_cpu_kernel(const std::map<int, std::map<std::string, std::vector<float>>>& tick_layer_weight, size_t starting_index)
{
    std::map<int, std::map<std::string, float>> output;

    auto starting_tick_iter = tick_layer_weight.begin();
    for (int i = 0; i < starting_index; ++i)
    {
        starting_tick_iter++;
    }
    int starting_tick = starting_tick_iter->first;
    const auto& starting_layer_weight = starting_tick_iter->second;
    for (const auto& [tick, layer_weight]: tick_layer_weight)
    {
        if (tick == starting_tick) continue;

        for (const auto& [layer_name, weight]: layer_weight)
        {
            float distance = 0;
            calculate_distance(&distance, starting_layer_weight.at(layer_name), weight);
            output[tick][layer_name] = distance;
        }
    }

    return output;
}

std::map<int, std::map<std::string, float>> calculate_model_distance_from_destination_cpu_kernel(const std::map<int, std::map<std::string, std::vector<float>>>& tick_layer_weight, size_t destination_index)
{
    std::map<int, std::map<std::string, float>> output;

    auto starting_tick_iter = tick_layer_weight.begin();
    for (int i = 0; i < destination_index; ++i)
    {
        starting_tick_iter++;
    }
    int destination_tick = starting_tick_iter->first;
    const auto& destination_layer_weight = starting_tick_iter->second;
    for (const auto& [tick, layer_weight]: tick_layer_weight)
    {
        if (tick == destination_tick) continue;

        for (const auto& [layer_name, weight]: layer_weight)
        {
            float distance = 0;
            calculate_distance(&distance, destination_layer_weight.at(layer_name), weight);
            output[tick][layer_name] = distance;
        }
    }

    return output;
}

std::map<int, std::map<std::string, float>> calculate_model_distance_from_origin_cpu_kernel(const std::map<int, std::map<std::string, std::vector<float>>>& tick_layer_weight)
{
    std::map<int, std::map<std::string, float>> output;

    for (const auto& [tick, layer_weight]: tick_layer_weight)
    {
        for (const auto& [layer_name, weight]: layer_weight)
        {
            float distance = 0;
            calculate_distance_to_origin(&distance, weight);
            output[tick][layer_name] = distance;
        }
    }

    return output;
}

std::map<int, std::map<std::string, float>> calculate_delta_model_weight_distance(const std::map<int, std::map<std::string, std::vector<float>>>& tick_layer_weight)
{
    std::map<int, std::map<std::string, float>> output;
    std::map<std::string, std::vector<float>> previous_layer_weight;
    for (const auto& [tick, layer_weight]: tick_layer_weight)
    {
        for (const auto& [layer_name, weight]: layer_weight)
        {
            auto iter_previous_layer_weight = previous_layer_weight.find(layer_name);
            if (iter_previous_layer_weight == previous_layer_weight.end())
            {
                previous_layer_weight.emplace(layer_name, weight);
            }
            else
            {
                float distance = 0;
                calculate_distance(&distance, weight, iter_previous_layer_weight->second);
                iter_previous_layer_weight->second = weight;
                output[tick][layer_name] = distance;
            }
        }
    }
    return output;
}

//// return: map< node_name,  pair< map<tick, distance_to_starting>, map<tick, distance_to_destination>, map<tick, distance_to_origin>, map<tick, delta_distance> > > >
using TYPE_TICK_NODE_DISTANCE = std::map<int, std::map<std::string, float>>;
std::map<std::string, std::tuple<TYPE_TICK_NODE_DISTANCE, TYPE_TICK_NODE_DISTANCE, TYPE_TICK_NODE_DISTANCE, TYPE_TICK_NODE_DISTANCE>> process_weight_distance_from_starting_point_and_origin_and_destination(std::map<std::string, std::map<int, std::filesystem::path>> node_name_tick_and_path, size_t starting_index, size_t destination_index, bool use_cuda)
{
    std::map<std::string, std::tuple<TYPE_TICK_NODE_DISTANCE, TYPE_TICK_NODE_DISTANCE, TYPE_TICK_NODE_DISTANCE, TYPE_TICK_NODE_DISTANCE>> output;
    std::mutex output_lock;

    boost::asio::thread_pool pool(std::thread::hardware_concurrency());
    for (const auto& [node_name, tick_and_path]:node_name_tick_and_path)
    {
        boost::asio::post(pool, [tick_and_path, node_name, &output_lock, &output, starting_index, destination_index, use_cuda](){
            std::map<int, std::map<std::string, float>> output_distance_from_starting, output_distance_from_destination, output_distance_from_origin, delta_model_wegiht_distance;
            std::map<int, std::map<std::string, std::vector<float>>> tick_layer_weight;
            {
                for (const auto &[tick, model_path]: tick_and_path)
                {
                    std::ifstream model_file;
                    model_file.open(model_path);
                    LOG_IF(FATAL, model_file.bad()) << "cannot open file: " << model_path.string();
                    std::stringstream buffer;
                    buffer << model_file.rdbuf();
                    auto model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<float>>(buffer.str());
                    std::map<std::string, std::vector<float>> layer_weight;
                    for (const auto &single_layer: model.getLayers())
                    {
                        const auto &layer_p = single_layer.getBlob_p();
                        if (!layer_p) continue;
                        const auto &data = layer_p->getData();
                        if (data.empty()) continue;
                        layer_weight.emplace(single_layer.getName(), data);
                    }
                    tick_layer_weight.emplace(tick, layer_weight);
                }
            }

            //check all models have the same size
            {
                bool first = true;
                std::map<std::string, size_t> size_of_each_layer;
                for (const auto &[tick, layer_weight]: tick_layer_weight)
                {
                    if (first)
                    {
                        first = false;
                        for (const auto &[layer_name, weight]: layer_weight)
                        {
                            size_of_each_layer[layer_name] = weight.size();
                        }
                    }
                    else
                    {
                        for (const auto &[layer_name, weight]: layer_weight)
                        {
                            LOG_IF(FATAL, size_of_each_layer[layer_name] != weight.size()) << "tick: " << tick << " has different weight size:" << weight.size() << " standard: " << size_of_each_layer[layer_name] << " in layer:" << layer_name;
                        }
                    }
                }
            }

            output_distance_from_starting = calculate_model_distance_from_starting_cpu_kernel(tick_layer_weight, starting_index);
            output_distance_from_destination = calculate_model_distance_from_destination_cpu_kernel(tick_layer_weight, destination_index);
            output_distance_from_origin = calculate_model_distance_from_origin_cpu_kernel(tick_layer_weight);
            delta_model_wegiht_distance = calculate_delta_model_weight_distance(tick_layer_weight);
            {
                std::lock_guard guard(output_lock);
                output[node_name] = std::make_tuple(output_distance_from_starting, output_distance_from_destination, output_distance_from_origin, delta_model_wegiht_distance);
            }
            LOG(INFO) << "finish calculating distance from starting point & origin for node: " << node_name;
        });
    }
    pool.join();

    return output;
}

void calculate_weight_distance_from_starting_point_and_origin_and_destination(const std::string& models_path_str, const std::string& output_path_str, bool use_cuda)
{
#pragma region Pre check input/output path
    std::filesystem::path models_path;
    {
        models_path.assign(models_path_str);
        if (!std::filesystem::exists(models_path))
        {
            LOG(FATAL) << models_path.string() << " doesn't exist";
        }
    }

    std::filesystem::path output_path;
    {
        output_path.assign(output_path_str);
        if (!std::filesystem::exists(output_path))
        {
            std::filesystem::create_directory(output_path);
        }
        output_path = output_path / "weight_distance";
        if (std::filesystem::exists(output_path))
        {
            //skip this procedure
            LOG(INFO) << "skip analyze model weight distance from starting point and origin";
            return;
        }
        else
        {
            std::filesystem::create_directory(output_path);
        }
    }
#pragma endregion

    LOG(INFO) << "processing analyzing model weight distance from starting point and origin , path: " << models_path.string();

    //// map < node_name , map < tick, path >
    std::map<std::string, std::map<int, std::filesystem::path>> node_name_tick_to_path;
    std::set<std::string> all_nodes;
    std::set<int> all_ticks;
    for (const auto & entry_0 : std::filesystem::directory_iterator(models_path))
    {
        if (!entry_0.is_directory()) continue;
        const auto& tick_folder = entry_0.path();
        const auto tick = tick_folder.filename();
        for (const auto & entry_1 : std::filesystem::directory_iterator(tick_folder))
        {
            if (!entry_1.is_regular_file()) continue;
            const auto& node_model_file = entry_1.path();
            const auto node_name = node_model_file.stem().string();

            int tick_int = std::stoi(tick);
            all_ticks.emplace(tick_int);
            all_nodes.emplace(node_name);
            node_name_tick_to_path[node_name][tick_int] = node_model_file;
        }
    }
    LOG(INFO) << "ticks to process: " << all_ticks.size() << ", nodes to process: " << all_nodes.size() << ", total: " << node_name_tick_to_path.size();

    int start_tick = *all_ticks.begin();
    int end_tick = *all_ticks.rbegin();
    auto write_to_file_data = process_weight_distance_from_starting_point_and_origin_and_destination(node_name_tick_to_path, start_tick, end_tick, use_cuda); //map is ordered, so the first tick is always located at 0

    ////write to files
    for (const auto& [node_name, all_distances]: write_to_file_data)
    {
        const auto& [distance_to_starting, distance_to_destination, distance_to_origin, delta_distance] = all_distances;

        const auto write_to_file = [&node_name, &output_path](const std::string& filename_prefix, const std::map<int, std::map<std::string, float>>& distances){
            std::ofstream file;
            auto output_file_path = output_path / (filename_prefix + node_name + ".csv");
            file.open(output_file_path);
            LOG_IF(FATAL, file.bad()) << "cannot open file " << output_file_path.string();

            //create header
            LOG_IF(FATAL, distances.empty()) << "no data (distance from start) in node: " << node_name;
            {
                file << "tick";
                for (const auto& [layer_name, _] : distances.begin()->second)
                {
                    file << "," << layer_name;
                }
                file << "\n";
            }

            for (const auto& [tick, layer_distance]: distances)
            {
                file << tick;
                for (const auto& [_, distance]: layer_distance)
                {
                    file << "," << distance;
                }
                file << "\n";
            }

            file.flush();
            file.close();
        };

        write_to_file("from_start_", distance_to_starting);
        write_to_file("from_destination_", distance_to_destination);
        write_to_file("from_origin_", distance_to_origin);
        write_to_file("delta_distance_", delta_distance);
    }

}