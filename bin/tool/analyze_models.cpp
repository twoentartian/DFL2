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
            ("fast_kernel,f", "use faster cuda kernel with more memory consumption")
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
    
    
    calculate_weight_distance_at_each_tick(models_path_str, output_path_str, use_cuda, use_faster_cuda_kernel);

    return 0;
}