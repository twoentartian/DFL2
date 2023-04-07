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


#include <measure_time.hpp>

#include "analyze_models_weight_distance_from_each_other.hpp"
#include "analyze_models_calculate_model_accuracy_detail.hpp"

using model_datatype = float;

int main(int argc, char** argv)
{
    std::string models_path_str;
    std::string output_path_str;
    std::string dataset_path_str;
    std::string solver_path_str;
    size_t test_size;
    boost::program_options::options_description desc("allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("model_path,m", boost::program_options::value<std::string>(&models_path_str)->default_value("./models"), "model folder path")
            ("output_path,o", boost::program_options::value<std::string>(&output_path_str)->default_value("./analyze_models_output"), "output folder path")
            ("cuda,c", "use cuda accelerator")
            ("fast_kernel,f", "use faster cuda kernel with more memory consumption")
            ("accuracy_test_size,s", boost::program_options::value<size_t>(&test_size)->default_value(200), "specify the tes batch size")
            ("no_mutual_distance", "disable: calculate mutual distance")
            ("no_accuracy_detail", "disable: calculate accuracy details")
            ("dataset_path", boost::program_options::value<std::string>(&dataset_path_str)->default_value("../../../dataset/MNIST/"), "specify dataset path")
            ("caffe_solver_path", boost::program_options::value<std::string>(&solver_path_str)->default_value("../../../dataset/MNIST/lenet_solver_memory.prototxt"), "specify the ML solver path")
            ;
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    bool use_cuda = false, use_faster_cuda_kernel = false;
    if (vm.count("cuda"))
    {
#if ANALYZE_MODEL_USE_CUDA
        use_cuda = true;
        LOG(INFO) << "use CUDA acceleration";
#else
        LOG(FATAL) << "CUDA is not supported when compiling";
#endif
    }
    
    if (vm.count("fast_kernel"))
    {
        use_faster_cuda_kernel = true;
    }
    else
    {
        use_faster_cuda_kernel = false;
    }
    
    
    //load dataset
    Ml::data_converter<model_datatype> train_dataset;
    Ml::data_converter<model_datatype> test_dataset;
    {
        std::filesystem::path dataset_path(dataset_path_str);
        LOG_IF(FATAL, !std::filesystem::exists(dataset_path)) << "dataset path does not exist";
        std::filesystem::path train_dataset_path = dataset_path / "train-images.idx3-ubyte";
        std::filesystem::path train_dataset_label_path = dataset_path / "train-labels.idx1-ubyte";
        std::filesystem::path test_dataset_path = dataset_path / "t10k-images.idx3-ubyte";
        std::filesystem::path test_dataset_label_path = dataset_path / "t10k-labels.idx1-ubyte";
        LOG_IF(FATAL, !std::filesystem::exists(train_dataset_path)) << train_dataset_path.string() << " does not exist";
        LOG_IF(FATAL, !std::filesystem::exists(train_dataset_label_path)) << train_dataset_label_path.string() << " does not exist";
        LOG_IF(FATAL, !std::filesystem::exists(test_dataset_path)) << test_dataset_path.string() << " does not exist";
        LOG_IF(FATAL, !std::filesystem::exists(test_dataset_label_path)) << test_dataset_label_path.string() << " does not exist";
        train_dataset.load_dataset_mnist(train_dataset_path, train_dataset_label_path);
        test_dataset.load_dataset_mnist(test_dataset_path, test_dataset_label_path);
    }
    
    if (!vm.count("no_mutual_distance"))
    {
        calculate_weight_distance_from_each_other(models_path_str, output_path_str, use_cuda, use_faster_cuda_kernel);
    }
    if (!vm.count("no_accuracy_detail"))
    {
        calculate_model_accuracy_details(models_path_str, output_path_str, train_dataset, test_dataset, solver_path_str, test_size);
    }


    return 0;
}