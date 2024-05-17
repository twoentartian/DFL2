#include <filesystem>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

int main(int argc, char *argv[]) {
    std::string csv_file_path;
    
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("path,p", po::value<std::string>(&csv_file_path)->required(), "network size, number of node")
            ;
    po::positional_options_description p;
    p.add("path", 1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }
    
    std::filesystem::path csv_path(csv_file_path);
    if (std::filesystem::exists(csv_path)) {
        std::cout << "processing: " << csv_file_path << std::endl;
    }
    else {
        std::cout << csv_file_path << " does not exist" << std::endl;
        return -1;
    }
    
    std::ifstream csv_file(csv_path, std::ios::binary);
    if (! csv_file.good() || ! csv_file.is_open()) {
        std::cout << "open " << csv_file_path << " failed" << std::endl;
        return -1;
    }
    
    
    
    
}