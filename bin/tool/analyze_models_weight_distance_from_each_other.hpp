#include <iostream>
#include <filesystem>
#include <map>
#include <mutex>

#include <glog/logging.h>
#include <boost/asio.hpp>

#include <ml_layer.hpp>
#include <boost_serialization_wrapper.hpp>

#include "analyze_models.hpp"

void calculate_distance(float *output, const std::vector<float> &data0, const std::vector<float> &data1)
{
    LOG_IF(FATAL, data0.size() != data1.size()) << "data sizes not equal";
    float v = 0;
    for (size_t index = 0; index < data0.size(); ++index)
    {
        auto t = data0[index] - data1[index];
        v += t * t;
    }
    *output = std::sqrt(v);
}


#if ANALYZE_MODEL_USE_CUDA

extern void cuda_malloc(void **device_ptr, size_t size);

void cuda_copy_device_memory_to_host(void *device_ptr, void *host_memory, size_t size);

void cuda_free(void *device_ptr);

extern void sync_all_cuda_stream();

extern void allocate_and_copy_device_memory(float **temp_device_ptr, const float *host_data, size_t size);

extern void run_kernel_2(float *lhs_device_data, float *rhs_device_data, float *output_device_data, size_t output_loc,
                         size_t output_size);

extern std::vector<float>
run_kernel(const std::vector<float> &weight_l, float *lhs_device_data, float *rhs_device_data);

extern void clear_gpu_memory(const std::map<std::string, float *> &node_layer_to_device_memory);

extern bool get_device_support_async_mem_management();

////*
/// return: map < <smaller_node, larger_node> : <layer_name : value> >
///
/// *////

std::map<std::pair<std::string, std::string>, std::map<std::string, float>>
calculate_model_distance_of_each_model_pair_gpu_kernel_with_fixed_memory(const std::map<std::string, std::map<std::string, std::vector<float>>> &node_layer_weight)
{
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> output;
    std::mutex output_lck;
    
    //allocate output
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end(); ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
        {
            if (iter_r == iter_l) continue;
            for (const auto &[layer_name, weight_l]: iter_l->second)
            {
                auto node_pair = std::make_pair(iter_l->first, iter_r->first);
                output[node_pair][layer_name] = 0;
            }
        }
    }
    
    std::map<std::string, float *> node_layer_to_device_memory;
    
    //copy layer weight to GPU
    {
        for (const auto &[node_name, layer_weight]: node_layer_weight)
        {
            for (const auto &[layer, weight]: layer_weight)
            {
                float *temp_device_ptr;
                allocate_and_copy_device_memory(&temp_device_ptr, weight.data(), weight.size() * sizeof(weight[0]));
                node_layer_to_device_memory.emplace(node_name + layer, temp_device_ptr);
            }
        }
    }
    
    boost::asio::thread_pool pool(4);
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end(); ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
        {
            if (iter_r == iter_l) continue;
            
            for (const auto &[layer_name, weight_l]: iter_l->second)
            {
                boost::asio::post(pool,[iter_l, iter_r, &node_layer_to_device_memory, &output, &output_lck, &layer_name, &weight_l]() {
                                      auto lhs_device_data_iter = node_layer_to_device_memory.find(iter_l->first + layer_name);
                                      if (lhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                                      float *lhs_device_data = lhs_device_data_iter->second;
                                      
                                      auto rhs_device_data_iter = node_layer_to_device_memory.find(iter_r->first + layer_name);
                                      if (rhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                                      float *rhs_device_data = rhs_device_data_iter->second;
                                      
                                      std::vector<float> host_buffer = run_kernel(weight_l, lhs_device_data,rhs_device_data);
                                      
                                      float v = 0;
                                      for (const auto &i: host_buffer)
                                      {
                                          v += i;
                                      }
                                      auto node_pair = std::make_pair(iter_l->first, iter_r->first);
                                      {
                                          std::lock_guard guard(output_lck);
                                          output.at(node_pair).at(layer_name) = std::sqrt(v);
                                      }
                                  });
                
            }
        }
    }
    pool.join();
    
    sync_all_cuda_stream();

    //clear gpu memory
    clear_gpu_memory(node_layer_to_device_memory);
    sync_all_cuda_stream();
    
    return output;
}

////*
/// return: map < <smaller_node, larger_node> : <layer_name : value> >
///
/// *////

std::map<std::pair<std::string, std::string>, std::map<std::string, float>>
calculate_model_distance_of_each_model_pair_gpu_kernel_hungry_for_memory(const std::map<std::string, std::map<std::string, std::vector<float>>> &node_layer_weight)
{
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> output;
    std::mutex output_lck;

    //allocate output
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end(); ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
        {
            if (iter_r == iter_l) continue;
            for (const auto &[layer_name, weight_l]: iter_l->second)
            {
                auto node_pair = std::make_pair(iter_l->first, iter_r->first);
                output[node_pair][layer_name] = 0;
            }
        }
    }
    
    std::map<std::string, float *> node_layer_to_device_memory;

    //copy layer weight to GPU
    {
        for (const auto &[node_name, layer_weight]: node_layer_weight)
        {
            for (const auto &[layer, weight]: layer_weight)
            {
                float *temp_device_ptr;
                allocate_and_copy_device_memory(&temp_device_ptr, weight.data(), weight.size() * sizeof(weight[0]));
                node_layer_to_device_memory.emplace(node_name + layer, temp_device_ptr);
            }
        }
        sync_all_cuda_stream();
    }
    
    
    boost::asio::thread_pool pool(4);
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end(); ++iter_l)
    {
        boost::asio::post(pool, [iter_l, &node_layer_weight, &node_layer_to_device_memory, &output_lck, &output]() {
            size_t total_calculation_count = node_layer_weight.size() * iter_l->second.size();
            std::vector<size_t> output_start_index;
            output_start_index.reserve(total_calculation_count);
            size_t output_size_byte = 0;
            size_t output_size = 0;
            std::vector<size_t> output_data_size_for_single_calculation;
            output_data_size_for_single_calculation.reserve(total_calculation_count);
            
            for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
            {
                if (iter_r == iter_l) continue;
                
                for (const auto &[layer_name, weight_r]: iter_r->second)
                {
                    output_start_index.push_back(output_size);
                    size_t temp_output_size_byte = sizeof(weight_r[0]) * weight_r.size();
                    output_size_byte += temp_output_size_byte;
                    output_data_size_for_single_calculation.push_back(weight_r.size());
                    output_size += weight_r.size();
                }
            }

            //allocate output memory
            float *device_output_ptr;
            cuda_malloc((void **) &device_output_ptr, output_size_byte);
            
            {
                size_t index = 0;
                for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
                {
                    if (iter_r == iter_l) continue;
                    
                    for (const auto &[layer_name, weight_l]: iter_l->second)
                    {
                        auto lhs_device_data_iter = node_layer_to_device_memory.find(iter_l->first + layer_name);
                        if (lhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                        float *lhs_device_data = lhs_device_data_iter->second;
                        
                        auto rhs_device_data_iter = node_layer_to_device_memory.find(iter_r->first + layer_name);
                        if (rhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                        float *rhs_device_data = rhs_device_data_iter->second;
                        
                        run_kernel_2(lhs_device_data, rhs_device_data, device_output_ptr, output_start_index[index],
                                     output_data_size_for_single_calculation[index]);
                        index++;
                    }
                }
            }

//wait for finish processing
            sync_all_cuda_stream();
            
            {
                std::vector<float> output_data;
                output_data.resize(output_size);
                cuda_copy_device_memory_to_host(device_output_ptr, output_data.data(), output_size_byte);
                
                size_t index = 0;
                for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
                {
                    if (iter_r == iter_l) continue;
                    
                    for (const auto &[layer_name, weight_l]: iter_l->second)
                    {
                        float v = 0;
                        for (size_t i = output_start_index[index];
                             i < output_data_size_for_single_calculation[index]; ++i)
                        {
                            v += output_data[i];
                        }
                        auto node_pair = std::make_pair(iter_l->first, iter_r->first);
                        {
                            std::lock_guard guard(output_lck);
                            output.at(node_pair).at(layer_name) = std::sqrt(v);
                        }
                    }
                }
            }
            cuda_free(device_output_ptr);
        });
    }
    pool.join();

//clear gpu memory
    clear_gpu_memory(node_layer_to_device_memory);
    
    return output;
}

#endif

////*
/// return: map < <smaller_node, larger_node> : <layer_name : value> >
///
/// *////
std::map<std::pair<std::string, std::string>, std::map<std::string, float>>
calculate_model_distance_of_each_model_pair_cpu_kernel(
        const std::map<std::string, std::map<std::string, std::vector<float>>> &node_layer_weight, int tick)
{
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> output;
    boost::asio::thread_pool pool(std::thread::hardware_concurrency());
    
    std::atomic_uint64_t finished_task = 0;
    std::atomic_uint32_t current_percentage = 0;
    size_t total_tasks =
            node_layer_weight.size() * (node_layer_weight.size() - 1) / 2 * node_layer_weight.begin()->second.size();
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end(); ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
        {
            if (iter_r == iter_l) continue;
            std::map<std::string, float> layer_weight_output;
            for (const auto &[layer_name, weight]: iter_l->second)
            {
                layer_weight_output.emplace(layer_name, 0);
            }
            output.emplace(std::make_pair(iter_l->first, iter_r->first), layer_weight_output);
            for (const auto &[layer_name, weight0]: iter_l->second)
            {
                const auto weight1_iter = iter_r->second.find(layer_name);
                const auto &weight1 = weight1_iter->second;
                auto &output_value_iter = output[{iter_l->first, iter_r->first}][layer_name];
                output_value_iter = 1;
                boost::asio::post(pool,
                                  [tick, total_tasks, &current_percentage, &finished_task, &output_value_iter, weight0, weight1]() {
                                      calculate_distance(&output_value_iter, weight0, weight1);
                                      finished_task++;
                                      auto temp_current_percentage = uint32_t(
                                              (float) finished_task / (float) total_tasks * 100);
                                      if (temp_current_percentage > current_percentage)
                                      {
                                          current_percentage = temp_current_percentage;
                                          {
                                              std::lock_guard guard(cout_mutex);
                                              std::cout << "processing tick " << tick << ":" << current_percentage
                                                        << "%" << std::endl;
                                          }
                                      }
                                  });
            }
        }
    }
    pool.join();
    
    return output;
}

std::map<std::pair<std::string, std::string>, std::map<std::string, float>> calculate_model_distance_of_each_model_pair(const std::map<std::string, std::filesystem::path> &node_name_and_model,bool use_cuda, bool use_faster_cuda_kernel, int tick)
{
    std::map<std::string, Ml::caffe_parameter_net<float>> models;
    for (const auto &[node_name, model_path]: node_name_and_model)
    {
        std::ifstream model_file;
        model_file.open(model_path);
        LOG_IF(FATAL, model_file.bad()) << "cannot open file: " << model_path.string();
        std::stringstream buffer;
        buffer << model_file.rdbuf();
        auto model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<float>>(buffer.str());
        models.emplace(node_name, std::move(model));
    }
    
    std::map<std::string, std::map<std::string, std::vector<float>>> node_layer_weight;
    for (const auto &[node_name, model]: models)
    {
        std::map<std::string, std::vector<float>> layer_weight;
        for (const auto &single_layer: model.getLayers())
        {
            const auto &layer_p = single_layer.getBlob_p();
            if (!layer_p) continue;
            const auto &data = layer_p->getData();
            if (data.empty()) continue;
            layer_weight.emplace(single_layer.getName(), data);
        }
        node_layer_weight.emplace(node_name, layer_weight);
    }
    
    //check all models have the same size
    {
        bool first = true;
        std::map<std::string, size_t> size_of_each_layer;
        for (const auto &[node_name, layer_weight]: node_layer_weight)
        {
            if (first)
            {
                first = false;
                for (const auto &[layer_name, weight]: layer_weight)
                {
                    size_of_each_layer[layer_name] = weight.size();
                }
            } else
            {
                for (const auto &[layer_name, weight]: layer_weight)
                {
                    LOG_IF(FATAL, size_of_each_layer[layer_name] != weight.size())
                                    << "node: " << node_name << " has different weight size:" << weight.size()
                                    << " standard: " << size_of_each_layer[layer_name] << " in layer:" << layer_name;
                }
            }
        }
    }

#if ANALYZE_MODEL_USE_CUDA
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> result;
    if (use_cuda)
    {
        static bool print_kernel_info = true;
        if (use_faster_cuda_kernel)
        {
            if (print_kernel_info)
            {
                print_kernel_info = false;
                LOG(INFO) << "use the faster GPU kernel with more memory consumption";
            }
            result = calculate_model_distance_of_each_model_pair_gpu_kernel_hungry_for_memory(node_layer_weight);
        }
        else
        {
            if (print_kernel_info)
            {
                print_kernel_info = false;
                LOG(INFO) << "use the GPU kernel with stable memory consumption";
            }
            result = calculate_model_distance_of_each_model_pair_gpu_kernel_with_fixed_memory(node_layer_weight);
        }
    }
    else
    {
        result = calculate_model_distance_of_each_model_pair_cpu_kernel(node_layer_weight, tick);
    }
#else
    auto result = calculate_model_distance_of_each_model_pair_cpu_kernel(node_layer_weight);
#endif
    
    return result;
}

void calculate_weight_distance_at_each_tick(const std::string& models_path_str, const std::string& output_path_str, bool use_cuda, bool use_faster_cuda_kernel)
{
    std::filesystem::path models_path;
    {
        models_path.assign(models_path_str);
        if (!std::filesystem::exists(models_path))
        {
            LOG(FATAL) << models_path.string() << " doesn't exist";
        }
        else
        {
            LOG(INFO) << "processing model path: " << models_path.string();
        }
    }
    
    std::filesystem::path output_path;
    {
        output_path.assign(output_path_str);
        if (!std::filesystem::exists(output_path))
        {
            std::filesystem::create_directory(output_path);
        }
        output_path = output_path / "weight_distance_from_each_other";
        if (!std::filesystem::exists(output_path))
        {
            std::filesystem::create_directory(output_path);
        }
    }
    
    std::map<std::string, std::filesystem::path> ticks_to_directories;
    for (const auto & entry : std::filesystem::directory_iterator(models_path))
    {
        if (entry.is_directory())
        {
            auto tick_folder_path = entry.path();
            ticks_to_directories.emplace(tick_folder_path.filename(), tick_folder_path);
        }
    }
    LOG(INFO) << "ticks to process: " << ticks_to_directories.size();
    
    ////calculate the model difference of each model pair in a tick folder
    std::map<int, std::map<std::pair<std::string, std::string>, std::map<std::string, float>>> tick_to_distance_result;
    
    for (const auto &[tick, tick_folder_path]: ticks_to_directories)
    {
        measure_time timer;
        timer.start();
        std::map<std::string, std::filesystem::path> node_and_model;
        for (const auto &entry: std::filesystem::directory_iterator(tick_folder_path))
        {
            if (entry.is_regular_file())
            {
                auto model_path = entry.path();
                node_and_model.emplace(model_path.stem().string(), model_path);
            }
        }
        auto distance_results = calculate_model_distance_of_each_model_pair(node_and_model, use_cuda, use_faster_cuda_kernel, std::stoi(tick));
        tick_to_distance_result.emplace(std::stoi(tick), distance_results);
        timer.stop();
        std::cout << "tick: " << tick << " costs " << timer.instant_measure_ms() << "ms" << std::endl;
    }
    
    ////save distance result to file
    ////*
    /// write_to_file_data:
    /// map < node_pair : < tick : <layer : distance > > >
    /// *////
    std::map<std::string, std::map<int, std::map<std::string, float>>> write_to_file_data;
    for (const auto& [tick, node_layer_distance] : tick_to_distance_result)
    {
        for (const auto& [node_pair, layer_distance]: node_layer_distance)
        {
            const auto& [node_0, node_1] = node_pair;
            std::stringstream ss;
            ss << node_0 << "-" << node_1;
            const std::string node_pair_identifier = ss.str();
            for (const auto& [layer_name, distance]: layer_distance)
            {
                write_to_file_data[node_pair_identifier][tick][layer_name] = distance;
            }
        }
    }
    
    ////write to files
    for (const auto& [node_pair_identifier, tick_layer_distance]: write_to_file_data)
    {
        std::ofstream file;
        auto output_file_path = output_path / (node_pair_identifier + ".csv");
        file.open(output_file_path);
        LOG_IF(FATAL, file.bad()) << "cannot open file " << output_file_path.string();
        
        //create header
        LOG_IF(FATAL, tick_layer_distance.empty()) << "no data in " << node_pair_identifier;
        {
            file << "tick";
            for (const auto& [layer_name, _] : tick_layer_distance.begin()->second)
            {
                file << "," << layer_name;
            }
            file << "\n";
        }
        
        for (const auto& [tick, layer_distance]: tick_layer_distance)
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
    }
}
