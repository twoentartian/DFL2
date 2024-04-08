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

using model_datatype = float;
namespace po = boost::program_options;

void generateCombinations(std::vector<std::vector<size_t>>& output, std::vector<size_t>& current, size_t sum, size_t max, size_t N) {
    if (current.size() == N) {
        if (sum <= max) {
            output.push_back(current); // Save the current combination
        }
        return; // Stop if we reach or exceed the sum limit
    }

    for (size_t i = 0; i <= max - sum; i ++) {
        current.push_back(i); // Add the current value to the combination
        generateCombinations(output, current, sum + i, max, N); // Recurse with the new sum
        current.pop_back(); // Backtrack to explore other possibilities
    }
}

//void generateCombinations(std::vector<std::vector<size_t>>& allCombinations, std::vector<size_t>& currentCombination, size_t max, size_t n) {
//    if (n == 0) {
//        // Base case: if n is 0, a combination of required length is formed
//        allCombinations.push_back(currentCombination);
//        return;
//    }
//
//    for (size_t i = 0; i <= max; i++) {
//        // Add current number to the combination
//        currentCombination.push_back(i);
//        size_t sum = 0;
//        for (const auto& v : currentCombination) {
//            sum += v;
//        }
//        if (sum <= max) {
//            // Recurse with n-1 to add the next element
//            generateCombinations(allCombinations, currentCombination, max, n - 1);
//        }
//        // Remove the last element to backtrack
//        currentCombination.pop_back();
//    }
//}

int main(int argc, char** argv) {
    setenv("OPENBLAS_NUM_THREADS", "1", 1);
    setenv("GOTO_NUM_THREADS", "1", 1);
    setenv("OMP_NUM_THREADS", "1", 1);

    std::mutex cout_mutex;

    std::vector<std::string> model_paths;
    std::string dataset_path_str;
    std::string solver_path_str;
    size_t resolution;
    size_t test_size;
    float max_fusion_ratio;
    bool use_fixed_test_dataset;
    po::options_description desc("allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("model_paths", po::value<std::vector<std::string>>(&model_paths)->multitoken()->required(), "model files")
            ("accuracy_test_size,s", po::value<size_t>(&test_size)->default_value(100), "specify the test batch size")
            ("resolution,r", po::value<size_t>(&resolution)->default_value(100), "specify the resolution as an integer: 100->0.01")
            ("dataset_path", po::value<std::string>(&dataset_path_str)->default_value("../../../dataset/MNIST/"), "specify dataset path")
            ("use_fixed_test_dataset,f", po::value<bool>(&use_fixed_test_dataset)->default_value(true), "use fixed testing dataset")
            ("caffe_solver_path", po::value<std::string>(&solver_path_str)->default_value("../../../dataset/MNIST/lenet_solver_memory.prototxt"), "specify the ML solver path")
            ("max_fusion_ratio,m", po::value<float>(&max_fusion_ratio)->default_value(1.0f), "the maximum fusion ratio. 2--the sum of all models=2xone model")
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

    //solver for testing
    Ml::MlCaffeModel<float, caffe::SGDSolver>* solver_for_testing;
    size_t solver_for_testing_size = std::thread::hardware_concurrency();
    solver_for_testing = new Ml::MlCaffeModel<float, caffe::SGDSolver>[solver_for_testing_size];
    for (int i = 0; i < solver_for_testing_size; ++i) {
        solver_for_testing[i].load_caffe_model(solver_path_str);
    }

    // generate fusion data
    std::cout << "generate fusion data" << std::endl;
    size_t number_of_model = all_models.size();
    std::vector<std::vector<float>> fusion_ratio;
    {
        std::vector<std::vector<size_t>> fusion_ratio_raw;
        std::vector<size_t> currentCombination;
        generateCombinations(fusion_ratio_raw, currentCombination, 0, size_t(resolution*max_fusion_ratio), number_of_model);

        fusion_ratio.reserve(fusion_ratio_raw.size());
        for (const auto& line : fusion_ratio_raw) {
            std::vector<float> proportion_line;
            proportion_line.reserve(line.size());
            float sum = 0.0f;
            for (const auto& v : line) {
                float vf = float(v)/float(resolution);
                proportion_line.push_back(vf);
                sum += vf;
            }
            if (sum <= max_fusion_ratio) {
                fusion_ratio.push_back(proportion_line);
            }
        }
    }
    std::cout << "generate fusion data done" << std::endl;

    // calculate accuracy
    std::atomic_uint64_t finished_task = 0;
    std::atomic_uint32_t current_percentage = 0;
    size_t total_tasks = fusion_ratio.size();
    LOG_ASSERT(!all_models.empty());
    LOG_ASSERT(all_model_names.size() == all_models.size());
    size_t model_size = all_models.size();
    std::map<std::vector<float>, float> accuracy_result, loss_result;
    std::mutex accuracy_result_lock;

    auto time_start = std::chrono::system_clock::now();
//    tmt::ParallelExecution_StepIncremental
    tmt::ParallelExecution_StepIncremental([total_tasks, &current_percentage, &time_start, &solver_for_testing, &finished_task, &cout_mutex, &test_dataset, &all_models, &test_size, &model_size, &fixed_test_dataset, &accuracy_result_lock, &accuracy_result, &loss_result](uint32_t index, uint32_t thread_index, const std::vector<float>& current_fusion_ratio) {
        Ml::caffe_parameter_net<model_datatype> fusion_model = all_models[0] * current_fusion_ratio[0];
        for (size_t i = 1; i < model_size; ++i) {
            auto temp = all_models[i] * current_fusion_ratio[i];
            fusion_model = fusion_model + temp;
        }

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

        {
            std::lock_guard guard(accuracy_result_lock);
            accuracy_result.emplace(current_fusion_ratio, accuracy);
            loss_result.emplace(current_fusion_ratio, loss);
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
    }, total_tasks, fusion_ratio.data());

    delete[] solver_for_testing;

    // save to file
    std::ofstream csv_file("fusion_model_accuracy.csv", std::ios::binary);
    LOG_ASSERT(csv_file.good());
    // header
    for (int i = 0; i < model_size; ++i) {
        csv_file << all_model_names[i] << ",";
    }
    csv_file << "accuracy,loss" << std::endl;

    // data
    for (const auto& single_ratio : fusion_ratio) {
        for (int i = 0; i < model_size; ++i) {
            csv_file << single_ratio[i] << ",";
        }
        csv_file << accuracy_result[single_ratio] << "," << loss_result[single_ratio] << std::endl;
    }

    csv_file.flush();
    csv_file.close();

    return 0;
}

