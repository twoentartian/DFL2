#pragma once

#include <map>

#include <ml_layer.hpp>
#include <boost_serialization_wrapper.hpp>

std::filesystem::path generate_output_file_path(const std::string& node_name, const std::filesystem::path& output_folder)
{
    return output_folder / (node_name + ".csv");
}

template<typename model_datatype>
void calculate_model_accuracy_details(const std::string& models_path_str, const std::string& output_path_str, Ml::data_converter<model_datatype>& train_dataset, Ml::data_converter<model_datatype>& test_dataset, const std::string& solver_path_str, size_t test_size)
{
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
        output_path = output_path / "accuracy_detail";
        if (std::filesystem::exists(output_path))
        {
            LOG(INFO) << "skip analyze model accuracy details";
            return;
        }
        else
        {
            std::filesystem::create_directory(output_path);
        }
    }
    
    LOG(INFO) << "processing analyzing model accuracy details, path: " << models_path.string();
    
    //// map < <tick, node_name> , path >
    std::map<std::pair<std::string, std::string>, std::filesystem::path> all_model_paths;
    for (const auto & entry_0 : std::filesystem::directory_iterator(models_path))
    {
        if (entry_0.is_directory())
        {
            for (const auto & entry_1 : std::filesystem::directory_iterator(entry_0))
            {
                const auto& path_0 = entry_0.path();
                const auto& path_1 = entry_1.path();
                all_model_paths.emplace(std::make_pair(path_0.filename(), path_1.stem().string()), path_1);
            }
        }
    }
    
    //// map <node_name, map < tick , map< label-test/train, path > > >
    std::map<std::string, std::map<std::string, std::map<std::string, float>>> output_accuracy_detail;
    std::mutex output_accuracy_detail_lock;
    
    {
        size_t concurrency_value = std::thread::hardware_concurrency(), concurrent_index = 0;
        auto* solver_for_testing = new Ml::MlCaffeModel<float, caffe::SGDSolver>[concurrency_value];
        for (int i = 0; i < concurrency_value; ++i)
        {
            solver_for_testing[i].load_caffe_model(solver_path_str);
        }
        boost::asio::thread_pool pool(concurrency_value);
        for (const auto& [tick_node_name, model_path]: all_model_paths)
        {
            const auto& [tick, node_name] = tick_node_name;
            const std::filesystem::path output_file_path = generate_output_file_path(node_name, output_path);
            if (std::filesystem::exists(output_file_path))
            {
                continue; // skip because the output is already generated
            }
            
            auto& output_node_tick = output_accuracy_detail[node_name][tick];
            
            boost::asio::post(pool,[solver_for_testing, concurrent_index, &model_path, &train_dataset, &test_dataset, &output_accuracy_detail_lock, &output_node_tick, tick, node_name, test_size](){
                auto& solver = solver_for_testing[concurrent_index];
                
                std::ifstream model_file;
                model_file.open(model_path);
                LOG_IF(FATAL, model_file.bad()) << "cannot open file: " << model_path.string();
                std::stringstream buffer;
                buffer << model_file.rdbuf();
                const auto model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<float>>(buffer.str());
                solver.set_parameter(model);
                
                for (int label = 0; label < 10; ++label) // for MNIST, the label max is 9
                {
                    Ml::tensor_blob_like<model_datatype> temp_label_tensor;
                    temp_label_tensor.getShape() = {1};
                    temp_label_tensor.getData() = {float(label)};
                    const auto [data_tensor, label_tensor] = train_dataset.get_random_data_by_Label(temp_label_tensor, test_size);
                    float accuracy = solver.evaluation(data_tensor, label_tensor);
                    {
                        std::lock_guard guard(output_accuracy_detail_lock);
                        output_node_tick["train" + std::to_string(label)] = accuracy;
                    }
                }
                
                for (int label = 0; label < 10; ++label) // for MNIST, the label max is 9
                {
                    Ml::tensor_blob_like<model_datatype> temp_label_tensor;
                    temp_label_tensor.getShape() = {1};
                    temp_label_tensor.getData() = {float(label)};
                    const auto [data_tensor, label_tensor] = test_dataset.get_random_data_by_Label(temp_label_tensor, test_size);
                    float accuracy = solver.evaluation(data_tensor, label_tensor);
                    {
                        std::lock_guard guard(output_accuracy_detail_lock);
                        output_node_tick["test" + std::to_string(label)] = accuracy;
                    }
                }
                
                LOG(INFO) << "finish tick: {" << tick << "} node name: {" << node_name << "}";
            });
            
            concurrent_index++;
            if (concurrent_index == concurrency_value) concurrent_index = 0;
            
        }
        pool.join();
        
        delete[] solver_for_testing;
    }
    
    ////write to files
    for (const auto& [node_name, other]: output_accuracy_detail)
    {
        std::ofstream file;
        const std::filesystem::path output_file_path = generate_output_file_path(node_name, output_path);
        file.open(output_file_path);
        LOG_IF(FATAL, file.bad()) << "cannot open file " << output_file_path.string();

        //create header
        {
            file << "tick";
            for (const auto& [column, _] : other.begin()->second)
            {
                file << "," << column;
            }
            file << "\n";
        }

        for (const auto& [tick, column_and_value]: other)
        {
            file << tick;
            for (const auto &[_, value]: column_and_value)
            {
                file << "," << value;
            }
            file << "\n";
        }
        file.flush();
        file.close();
    }

    
}