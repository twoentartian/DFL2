#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iostream>
#include <cstdlib>

#include <boost/program_options.hpp>
#include <boost/asio.hpp>

#include <tmt.hpp>
#include <ml_layer.hpp>
#include "../simulation/simulator_opti_model_update.hpp"

using model_datatype = float;
namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::mutex cout_mutex;

    std::vector<std::string> model_paths;
    std::string dataset_path_str;
    std::string solver_path_str;
    size_t test_size;
    bool use_fixed_test_dataset;
    po::options_description desc("allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("model_paths", po::value<std::vector<std::string>>(&model_paths)->multitoken()->required(), "model files")
            ("accuracy_test_size,s", po::value<size_t>(&test_size)->default_value(100), "specify the test batch size")
            ("dataset_path", po::value<std::string>(&dataset_path_str)->default_value("../../../dataset/MNIST/"), "specify dataset path")
            ("use_fixed_test_dataset,f", po::value<bool>(&use_fixed_test_dataset)->default_value(true), "use fixed testing dataset")
            ("caffe_solver_path", po::value<std::string>(&solver_path_str)->default_value("../../../dataset/MNIST/lenet_solver_memory.prototxt"), "specify the ML solver path")
            ;
    po::positional_options_description p;
    p.add("model_paths", -1); // Allowing all positional arguments to be treated as "compulsory"

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    //load dataset
    Ml::data_converter<model_datatype> test_dataset;
    std::optional<std::tuple<std::vector<const Ml::tensor_blob_like<model_datatype>*>, std::vector<const Ml::tensor_blob_like<model_datatype>*>>> fixed_test_dataset;
    {
        std::filesystem::path dataset_path(dataset_path_str);
        LOG_IF(FATAL, !std::filesystem::exists(dataset_path)) << "dataset path does not exist";
        std::filesystem::path test_dataset_path = dataset_path / "t10k-images.idx3-ubyte";
        std::filesystem::path test_dataset_label_path = dataset_path / "t10k-labels.idx1-ubyte";
        LOG_IF(FATAL, !std::filesystem::exists(test_dataset_path)) << test_dataset_path.string() << " does not exist";
        LOG_IF(FATAL, !std::filesystem::exists(test_dataset_label_path)) << test_dataset_label_path.string() << " does not exist";
        test_dataset.load_dataset_mnist(test_dataset_path, test_dataset_label_path);

        if (use_fixed_test_dataset) {
            std::vector<const Ml::tensor_blob_like<model_datatype>*> all_test_data, all_test_label;
            const std::vector<Ml::tensor_blob_like<model_datatype>>& all_labels = test_dataset.get_label();

            const std::unordered_map<std::string, std::vector<Ml::tensor_blob_like<model_datatype>>>& data_by_label = test_dataset.get_container_by_label();
            std::map<std::string, const Ml::tensor_blob_like<model_datatype>*> all_label_types;
            for (const auto& single_label : all_labels) {
                const std::string single_label_digest = single_label.get_str();
                if (!all_label_types.contains(single_label_digest)) all_label_types.emplace(single_label_digest, &single_label);
            }

            const size_t single_label_sample_size = test_size / all_label_types.size();
            LOG_IF(FATAL, single_label_sample_size*all_label_types.size() != test_size) << "test batch size % all_labels.size() != 0";
            for (const auto& [single_label_digest, single_label] : all_label_types) {
                const std::vector<Ml::tensor_blob_like<model_datatype>>& all_data_of_label = data_by_label.at(single_label_digest);
                for (size_t i = 0; i < single_label_sample_size; ++i) {
                    all_test_data.push_back(&all_data_of_label[i]);
                    all_test_label.push_back(single_label);
                }
            }
            fixed_test_dataset = {all_test_data, all_test_label};
        }
    }

    // load model
    std::vector<Ml::caffe_parameter_net<model_datatype>> all_models;
    std::vector<std::string> all_model_names;
    for (const auto& model_path_str : model_paths) {
        std::filesystem::path model_path(model_path_str);
        if (!std::filesystem::exists(model_path)) {
            LOG(FATAL) << model_path << " does not exist";
        }

        if (std::filesystem::is_directory(model_path)) {
            for (const auto & entry : std::filesystem::directory_iterator(model_path)) {
                const std::filesystem::path& _model_path(entry);
                std::ifstream model_file;
                model_file.open(_model_path, std::ios::binary);
                LOG_IF(FATAL, model_file.bad()) << "cannot open file: " << _model_path;
                std::stringstream buffer;
                buffer << model_file.rdbuf();
                auto model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<float>>(buffer.str());
                std::string model_name = _model_path.filename().replace_extension();
                all_models.emplace_back(model);
                all_model_names.push_back(model_name);
                std::cout << "loading model: " << model_name << ". with path: " << _model_path << std::endl;
            }
        }
        else if (std::filesystem::is_regular_file(model_path)) {
            std::ifstream model_file;
            model_file.open(model_path, std::ios::binary);
            LOG_IF(FATAL, model_file.bad()) << "cannot open file: " << model_path;
            std::stringstream buffer;
            buffer << model_file.rdbuf();
            auto model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<float>>(buffer.str());
            std::string model_name = model_path.filename().replace_extension();
            all_models.emplace_back(model);
            all_model_names.push_back(model_name);
            std::cout << "loading model: " << model_name << ". with path: " << model_path << std::endl;
        }
    }

    //solver for testing
    Ml::MlCaffeModel<float, caffe::SGDSolver>* solver_for_testing;
    size_t solver_for_testing_size = std::thread::hardware_concurrency();
    solver_for_testing = new Ml::MlCaffeModel<float, caffe::SGDSolver>[solver_for_testing_size];
    for (int i = 0; i < solver_for_testing_size; ++i) {
        solver_for_testing[i].load_caffe_model(solver_path_str);
    }

    // generate model pairs
    std::cout << "generate model pair" << std::endl;
    size_t number_of_model = all_models.size();
    std::vector<std::pair<size_t, size_t>> all_model_pairs;
    {
        std::set<std::pair<size_t, size_t>> all_model_pairs_set;
        for (size_t i = 0; i < all_models.size(); ++i) {
            for (size_t j = 0; j < all_models.size(); ++j) {
                if (i==j) {
                    continue;
                }
                if (i<j) {
                    all_model_pairs_set.emplace(i,j);
                }
                else {
                    all_model_pairs_set.emplace(j, i);
                }
            }
        }
        for (const auto& i : all_model_pairs_set) {
            all_model_pairs.push_back(i);
        }
    }
    std::cout << "generate model pair done" << std::endl;

    // calculate accuracy
    std::atomic_uint64_t finished_task = 0;
    std::atomic_uint32_t current_percentage = 0;
    size_t total_tasks = all_model_pairs.size();
    LOG_ASSERT(!all_models.empty());
    LOG_ASSERT(all_model_names.size() == all_models.size());
    size_t model_size = all_models.size();
    std::map<std::pair<size_t, size_t>, float> accuracy_result, accuracy_after_vc_result, loss_result, loss_after_vc_result, distance_result, distance_result_square;
    std::mutex accuracy_result_lock;

    auto time_start = std::chrono::system_clock::now();
    tmt::ParallelExecution_StepIncremental([total_tasks, &current_percentage, &time_start, &solver_for_testing, &finished_task, &cout_mutex, &test_dataset, &all_models, &test_size, &fixed_test_dataset, &accuracy_result_lock, &accuracy_result, &distance_result, &distance_result_square, &loss_result, &accuracy_after_vc_result, &loss_after_vc_result](uint32_t index, uint32_t thread_index, const std::pair<size_t,size_t>& current_model_pair) {
        auto [index0, index1] = current_model_pair;
        Ml::caffe_parameter_net<model_datatype> fusion_model = all_models[index0] * 0.5 + all_models[index1] * 0.5;
        Ml::caffe_parameter_net<model_datatype> fusion_model_after_vc = fusion_model.deep_clone();
        std::map<std::string, model_datatype> self_variance = opti_model_update_util::get_variance_for_model(fusion_model);
        {
            auto var_model1 = opti_model_update_util::get_variance_for_model(all_models[index0]);
            {
                auto var_model2 = opti_model_update_util::get_variance_for_model(all_models[index1]);
                for (auto& [k, v] : var_model1) {
                    v += var_model2[k];
                }
                for (auto& [k, v] : var_model1) {
                    v /= 2;
                }
            }
            for (Ml::caffe_parameter_layer<model_datatype>& layer : fusion_model_after_vc.getLayers()) {
                const std::string& name = layer.getName();
                const auto &blobs = layer.getBlob_p();
                if (!blobs.empty()) {
                    opti_model_update_util::scale_variance(blobs[0]->getData(), var_model1[name], 1.0f, self_variance[name]);
                }
            }
        }

        auto distance_model = all_models[index0] - all_models[index1];
        distance_model.abs();
        auto distance = distance_model.sum();
    
        auto distance_square_model = all_models[index0].dot_product(all_models[index0]) - all_models[index1].dot_product(all_models[index1]);
        distance_square_model.abs();
        auto distance_square = distance_square_model.sum();
        
        //testing
        std::vector<const Ml::tensor_blob_like<model_datatype>*> test_data, test_label;
        if (fixed_test_dataset.has_value()) {
            std::tie(test_data, test_label) = *fixed_test_dataset;
        }
        else {
            std::tie(test_data, test_label) = test_dataset.get_random_data(test_size);
        }

        solver_for_testing[thread_index].set_parameter(fusion_model);
        float loss = 0;
        auto accuracy = solver_for_testing[thread_index].evaluation(test_data, test_label, &loss);

        solver_for_testing[thread_index].set_parameter(fusion_model_after_vc);
        float loss_after_vc = 0;
        auto accuracy_after_vc = solver_for_testing[thread_index].evaluation(test_data, test_label, &loss_after_vc);

        {
            std::lock_guard guard(accuracy_result_lock);
            distance_result.emplace(current_model_pair, distance);
            distance_result_square.emplace(current_model_pair, distance_square);
            accuracy_result.emplace(current_model_pair, accuracy);
            loss_result.emplace(current_model_pair, loss);
            accuracy_after_vc_result.emplace(current_model_pair, accuracy_after_vc);
            loss_after_vc_result.emplace(current_model_pair, loss_after_vc);
        }

        finished_task++;

        const size_t total_ticks = 1000;
        auto temp_current_percentage = uint32_t((float) finished_task / (float) total_tasks * total_ticks);
        if (temp_current_percentage > current_percentage)
        {
            current_percentage = temp_current_percentage;
            auto time_now = std::chrono::system_clock::now();
            auto time_elapsed = time_now - time_start;
            time_start = time_now;
            auto remaining_time = (total_ticks - current_percentage) * time_elapsed;
            auto end_time = time_start + remaining_time;
            std::time_t end_time_c = std::chrono::system_clock::to_time_t(end_time);
            std::tm* end_time_tm = std::gmtime(&end_time_c);
            {
                std::lock_guard guard(cout_mutex);
                std::cout << "finishing " << current_percentage << "%%," << " expected to finish: " << std::put_time(end_time_tm, "%Y-%m-%d %H:%M:%S")<< std::endl;
            }
        }
    }, all_model_pairs.size(), all_model_pairs.data());

    delete[] solver_for_testing;

    // save to file
    {
        std::ofstream csv_file("model_similarity.csv", std::ios::binary);
        LOG_ASSERT(csv_file.good());
        // header
        csv_file << "node_index0,node_index1,accuracy,loss,accuracy_after_vc,loss_after_vc,distance,distance_square" << std::endl;
        // data
        for (const auto &single_pair: all_model_pairs) {
            auto [index0, index1] = single_pair;
            csv_file << all_model_names[index0] << "," << all_model_names[index1] << ",";
            csv_file << accuracy_result[single_pair] << "," << loss_result[single_pair] << ",";
            csv_file << accuracy_after_vc_result[single_pair] << "," << loss_after_vc_result[single_pair] << ",";
            csv_file << distance_result[single_pair] << "," << distance_result_square[single_pair] << std::endl;
        }

        csv_file.flush();
        csv_file.close();
    }

    // save to net file
    {
        std::ofstream net_file("model_similarity.net", std::ios::binary);
        LOG_ASSERT(net_file.good());

        // header
        net_file << "*Vertices" << " " << all_models.size() << std::endl;
        for (size_t i=0; i < all_model_names.size(); i++) {
            net_file << i+1 << " " << "\"" << all_model_names[i] << "\"" << std::endl;
        }

        // data
        net_file << "*Arcs" << std::endl;
        for (const auto &single_pair: all_model_pairs) {
            auto [index0, index1] = single_pair;
            net_file << index0+1 << " " << index1+1 << " " << loss_result[single_pair] << std::endl;
        }

        net_file.flush();
        net_file.close();
    }

    return 0;
}

