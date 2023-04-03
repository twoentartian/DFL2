#include <vector>
#include <iostream>
#include <filesystem>
#include <map>
#include <fstream>
#include <sstream>
#include <mutex>

#include <glog/logging.h>
#include <boost/asio.hpp>
#include <boost/program_options.hpp>

#include <ml_layer.hpp>
#include <measure_time.hpp>
#include <boost_serialization_wrapper.hpp>

std::mutex cout_mutex;

void calculate_distance(float* output, const std::vector<float>& data0, const std::vector<float>& data1)
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
extern void sync_all_cuda_stream();

extern void allocate_and_copy_device_memory(float** temp_device_ptr, const float* host_data, size_t size);

extern std::vector<float> run_kernel(const std::vector<float>& weight_l, float* lhs_device_data, float* rhs_device_data);

extern void clear_gpu_memory(const std::map<std::string, float*>& node_layer_to_device_memory);

////*
/// return: map < <smaller_node, larger_node> : <layer_name : value> >
///
/// *////

std::map<std::pair<std::string, std::string>, std::map<std::string, float>> calculate_model_distance_of_each_model_pair_gpu_kernel(const std::map<std::string, std::map<std::string, std::vector<float>>>& node_layer_weight)
{
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> output;
    std::mutex output_lck;
    
    //allocate output
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end() ; ++iter_l)
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
    
    std::map<std::string, float*> node_layer_to_device_memory;
    
    //copy layer weight to GPU
    {
        for (const auto& [node_name, layer_weight] : node_layer_weight)
        {
            for (const auto& [layer, weight] : layer_weight)
            {
                float* temp_device_ptr;
                allocate_and_copy_device_memory(&temp_device_ptr, weight.data(), weight.size() * sizeof(weight[0]));
                node_layer_to_device_memory.emplace(node_name+layer, temp_device_ptr);
            }
        }
    }
    
    
    std::vector<std::thread> pools;
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end() ; ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
        {
            if (iter_r == iter_l) continue;
            
            for (const auto& [layer_name, weight_l] : iter_l->second)
            {
                std::thread temp_thread([iter_l, iter_r, &node_layer_to_device_memory, &output, &output_lck, &layer_name, &weight_l](){
                    auto lhs_device_data_iter = node_layer_to_device_memory.find(iter_l->first + layer_name);
                    if (lhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                    float* lhs_device_data = lhs_device_data_iter->second;
                    
                    auto rhs_device_data_iter = node_layer_to_device_memory.find(iter_r->first + layer_name);
                    if (rhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                    float* rhs_device_data = rhs_device_data_iter->second;
                    
                    std::vector<float> host_buffer = run_kernel(weight_l, lhs_device_data, rhs_device_data);
                    
                    float v = 0;
                    for (const auto& i: host_buffer)
                    {
                        v += i;
                    }
                    auto node_pair = std::make_pair(iter_l->first, iter_r->first);
                    
                    {
                        std::lock_guard guard(output_lck);
                        output.at(node_pair).at(layer_name) = std::sqrt(v);
                    }
                });
                
                std::thread dummy;
                dummy.swap(temp_thread);
                pools.push_back(std::move(dummy));
            }
        }
    }
    for (auto& thread: pools)
    {
        thread.join();
    }
    
    sync_all_cuda_stream();
    
    //clear gpu memory
    clear_gpu_memory(node_layer_to_device_memory);
    sync_all_cuda_stream();
    
    return output;
}
#endif

////*
/// return: map < <smaller_node, larger_node> : <layer_name : value> >
///
/// *////
std::map<std::pair<std::string, std::string>, std::map<std::string, float>> calculate_model_distance_of_each_model_pair_cpu_kernel(const std::map<std::string, std::map<std::string, std::vector<float>>>& node_layer_weight, int tick)
{
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> output;
    boost::asio::thread_pool pool(4);
    
    std::atomic_uint64_t finished_task = 0;
    std::atomic_uint32_t current_percentage = 0;
    size_t total_tasks = node_layer_weight.size() * (node_layer_weight.size()-1) / 2 * node_layer_weight.begin()->second.size();
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end() ; ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end() ; ++iter_r)
        {
            if (iter_r == iter_l) continue;
            std::map<std::string, float> layer_weight_output;
            for (const auto& [layer_name, weight]: iter_l->second)
            {
                layer_weight_output.emplace(layer_name, 0);
            }
            output.emplace(std::make_pair(iter_l->first, iter_r->first), layer_weight_output);
            for (const auto& [layer_name, weight0]: iter_l->second)
            {
                const auto weight1_iter = iter_r->second.find(layer_name);
                const auto& weight1 = weight1_iter->second;
                auto& output_value_iter = output[{iter_l->first, iter_r->first}][layer_name];
                output_value_iter = 1;
                boost::asio::post(pool, [tick, total_tasks, &current_percentage, &finished_task, &output_value_iter, weight0, weight1](){
                    calculate_distance(&output_value_iter, weight0, weight1);
                    finished_task++;
                    auto temp_current_percentage = uint32_t((float)finished_task / (float)total_tasks * 100);
                    if (temp_current_percentage > current_percentage)
                    {
                        current_percentage = temp_current_percentage;
                        {
                            std::lock_guard guard(cout_mutex);
                            std::cout << "processing tick " << tick << ":" << current_percentage << "%" << std::endl;
                        }
                    }
                });
            }
        }
    }
    pool.join();
    
    return output;
}

std::map<std::pair<std::string, std::string>, std::map<std::string, float>> calculate_model_distance_of_each_model_pair(const std::map<std::string, std::filesystem::path>& node_name_and_model, bool use_cuda, int tick)
{
    std::map<std::string, Ml::caffe_parameter_net<float>> models;
    for (const auto& [node_name, model_path]: node_name_and_model)
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
    for (const auto& [node_name, model]: models)
    {
        std::map<std::string, std::vector<float>> layer_weight;
        for (const auto& single_layer: model.getLayers())
        {
            const auto& layer_p = single_layer.getBlob_p();
            if (!layer_p) continue;
            const auto& data = layer_p->getData();
            if (data.empty()) continue;
            layer_weight.emplace(single_layer.getName(), data);
        }
        node_layer_weight.emplace(node_name, layer_weight);
    }
    
    //check all models have the same size
    {
        bool first = true;
        std::map<std::string, size_t> size_of_each_layer;
        for (const auto& [node_name, layer_weight]: node_layer_weight)
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
                    LOG_IF(FATAL, size_of_each_layer[layer_name] != weight.size()) << "node: " << node_name << " has different weight size:" << weight.size() << " standard: " << size_of_each_layer[layer_name] << " in layer:" << layer_name;
                }
            }
        }
    }

#if ANALYZE_MODEL_USE_CUDA
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> result;
    if (use_cuda)
    {
        result = calculate_model_distance_of_each_model_pair_gpu_kernel(node_layer_weight);
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


int main(int argc, char** argv)
{
    std::string models_path_str;
    std::string output_path_str;
    boost::program_options::options_description desc("allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("model_path,m", boost::program_options::value<std::string>(&models_path_str)->default_value("./models"), "model folder path")
            ("output_path,o", boost::program_options::value<std::string>(&output_path_str)->default_value("./analyze_models_output"), "output folder path")
            ("cuda,c", "use cuda accelerator")
            ;
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    bool use_cuda = false;
    if (vm.count("cuda"))
    {
#if ANALYZE_MODEL_USE_CUDA
        use_cuda = true;
        LOG(INFO) << "use CUDA acceleration";
#else
        LOG(FATAL) << "CUDA is not supported when compiling";
#endif
    }
    
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
        auto distance_results = calculate_model_distance_of_each_model_pair(node_and_model, use_cuda, std::stoi(tick));
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

    return 0;
//    constexpr uint32_t size = 100000000;
//    std::vector<float> data0;
//    std::vector<float> data1;
//    data0.resize(size);
//    data1.resize(size);
//    for (int i = 0; i < size; ++i)
//    {
//        data0[i] = 0.0;
//        data1[i] = 0.5;
//    }
//
//    auto output = calculate_square_of_value_host(data0, data1);
//    for (const auto& v: output)
//    {
//        std::cout << v << " ";
//    }
//    std::cout << std::endl;
}