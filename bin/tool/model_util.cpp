#include <iostream>
#include <map>
#include <filesystem>
#include <optional>

#include <boost/program_options.hpp>

#include <glog/logging.h>
#include <tmt.hpp>
#include <ml_layer.hpp>
#include <boost_serialization_wrapper.hpp>

using model_datatype = float;

struct optional_args {
    std::optional<std::string> output_path;
    std::optional<std::string> output_layer_name;
};

class command_proto {
public:
    virtual std::string get_name() = 0;

    virtual std::tuple<int, std::string> execute(const std::map<int, std::string>&, const optional_args&) = 0;
};

class command_interpreter {
private:
    std::map<std::string, std::shared_ptr<command_proto>> commands;
    static command_interpreter* _instance;
    command_interpreter() = default;;
public:
    static command_interpreter* instance() {
        if (_instance == nullptr) {
            static command_interpreter temp_instance;
            _instance = &temp_instance;
        }
        return _instance;
    }

    void add_command(const std::shared_ptr<command_proto>& command_inst) {
        const std::string& name = command_inst->get_name();
        if (commands.find(name) != commands.end()) {
            throw std::logic_error("command already exists");
        }
        commands[name] = command_inst;
    }

    std::tuple<int, std::string> execute(const std::string& command, const std::map<int, std::string>& args, const optional_args& optional_args) {
        const auto command_inst = commands.find(command);
        if (command_inst == commands.end()) {
            return {1, "command not found"};
        }
        const auto [rc, info] = command_inst->second->execute(args, optional_args);
        return {rc, info};
    }
};
command_interpreter* command_interpreter::_instance = nullptr;

std::map<std::string, std::filesystem::path> func_list_file_in_dir(std::filesystem::path& path) {
    std::map<std::string, std::filesystem::path> output;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            auto filename = entry.path().filename();
            output[filename] = entry.path();
        }
    }
    return output;
}

class command_sub : public command_proto {
private:
    const std::string command = "sub";

public:
    std::string get_name() override {
        return command;
    }

    std::tuple<int, std::string> execute(const std::map<int, std::string>& args, const optional_args& optional_args) override {
        if (args.size() != 2) return {1, "sub has two args x and y, performing x-y"};

        std::filesystem::path arg0_path = (args.at(0)), arg1_path = (args.at(1));
        if (!exists(arg0_path)) return {1, arg0_path.string() + " does not exist"};
        if (!exists(arg1_path)) return {1, arg1_path.string() + " does not exist"};

        auto all_files_0 = func_list_file_in_dir(arg0_path);
        auto all_files_1 = func_list_file_in_dir(arg1_path);

        std::filesystem::path output_path;
        if (!optional_args.output_path->empty())
            output_path = *optional_args.output_path;
        else
            output_path = std::filesystem::current_path() / (arg0_path.filename().string() + "-" + arg1_path.filename().string());
        if (!exists(output_path)) std::filesystem::create_directories(output_path);

        //check all files in 0 are also in 1
        std::vector<std::string> all_names;
        for (const auto& [name, file_path] : all_files_0) {
            if (!all_files_1.contains(name)) return {-1, name + " does not exist in " + arg1_path.string()};
            all_names.push_back(name);
        }

        //load models
        tmt::ParallelExecution([&all_files_0, &all_files_1, &output_path](uint32_t index, uint32_t thread_index, const std::string& name){
            auto file_path0 = all_files_0[name];
            auto file_path1 = all_files_1[name];

            std::stringstream ss0, ss1;
            {
                std::ifstream fs0(file_path0, std::ios::binary);
                LOG_IF(FATAL, !fs0) << " cannot open " + file_path0.string();
                ss0 << fs0.rdbuf();
                fs0.close();
            }
            {
                std::ifstream fs1(file_path1, std::ios::binary);
                LOG_IF(FATAL, !fs1) << " cannot open " + file_path1.string();
                ss1 << fs1.rdbuf();
                fs1.close();
            }
            Ml::caffe_parameter_net<model_datatype> model0 = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<model_datatype>> (ss0);
            Ml::caffe_parameter_net<model_datatype> model1 = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<model_datatype>> (ss1);
            Ml::caffe_parameter_net<model_datatype> result_model = model0 - model1;

            std::ofstream fs_output(output_path/name, std::ios::binary);
            LOG_IF(FATAL, !fs_output) << "cannot write to " + (output_path/name).string();
            fs_output << serialize_wrap<boost::archive::binary_oarchive>(result_model).rdbuf();
            fs_output.flush();
            fs_output.close();
        }, all_names.size(), all_names.data());

        return {0, ""};
    }
};


class command_to_csv : public command_proto {
private:
    const std::string command = "to_csv";

public:
    std::string get_name() override {
        return command;
    }

    std::tuple<int, std::string> execute(const std::map<int, std::string>& args, const optional_args& optional_args) override {
        if (args.size() != 1) return {1, "to_csv has one arg x, converting layer l of model x to csv file, pass the layer name to -l"};

        std::filesystem::path model_path = (args.at(0));
        std::filesystem::path output_path = model_path;
        output_path.replace_extension(".csv");
        std::map<std::string, std::filesystem::path> all_models = func_list_file_in_dir(model_path);
        std::map<int, std::filesystem::path> tick_to_models;
        for (const auto& [name, path] : all_models) {
            int tick = std::stoi(name);
            tick_to_models[tick] = path;
        }

        std::ofstream output_csv(output_path, std::ios::binary);
        LOG_IF(FATAL, !output_csv) << " cannot write to " + output_path.string();
        ////create header
        {
            std::stringstream ss;
            {
                std::ifstream fs(tick_to_models.begin()->second, std::ios::binary);
                LOG_IF(FATAL, !fs) << " cannot open " + tick_to_models.begin()->second.string();
                ss << fs.rdbuf();
                fs.close();
            }
            Ml::caffe_parameter_net<model_datatype> model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<model_datatype>> (ss);
            output_csv << "tick" << "," << "type";
            for (const Ml::caffe_parameter_layer<model_datatype>& layer : model.getLayers()) {
                const auto& layer_name = layer.getName();
                const size_t layer_size = layer.size();
                if (layer_size == 0) continue;
                std::cout << "finds layer name: " << layer_name << ", size: " << layer_size << std::endl;
                if (optional_args.output_layer_name->empty()) continue;
                if (*optional_args.output_layer_name != layer_name) continue;

                for (size_t j = 0; j < layer_size; ++j) {
                    output_csv << "," << layer_name + "-" + std::to_string(j);
                }
                output_csv << "," << "distance" << "+" << layer_name;
                for (size_t j = 0; j < layer_size; ++j) {
                    output_csv << "," << layer_name + "-" + std::to_string(j) + "-" + "angle";
                }
            }
            output_csv << std::endl;
        }
        ////create content
        for (const auto& [tick, _model_path] : tick_to_models) {
            std::stringstream ss;
            {
                std::ifstream fs(_model_path, std::ios::binary);
                LOG_IF(FATAL, !fs) << " cannot open " + _model_path.string();
                ss << fs.rdbuf();
                fs.close();
            }
            Ml::caffe_parameter_net<model_datatype> model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<model_datatype>> (ss);

            output_csv << tick << "," << "STATUS";
            for (const Ml::caffe_parameter_layer<model_datatype>& layer : model.getLayers()) {
                const auto& layer_name = layer.getName();
                const auto layer_size = layer.size();
                if (layer_size == 0) continue;
                if (optional_args.output_layer_name->empty()) continue;
                if (*optional_args.output_layer_name != layer_name) continue;
                const auto& data = layer.getBlob_p()->getData();
                model_datatype distance_sum = 0;
                for (const auto& v : data) {
                    output_csv << "," << v;
                    distance_sum += v*v;
                }
                //store distance
                model_datatype distance = std::sqrt(distance_sum);
                output_csv << "," << distance;
                //calculate angle
                for (const auto& v : data) {
                    output_csv << "," << v / distance;
                }
            }
            output_csv << std::endl;
        }

        return {0, ""};
    }
};

int main(int argc, char** argv) {
    ////add command
    command_interpreter::instance()->add_command(std::static_pointer_cast<command_proto>(std::make_shared<command_sub>()));
    command_interpreter::instance()->add_command(std::static_pointer_cast<command_proto>(std::make_shared<command_to_csv>()));

    namespace po = boost::program_options;

    po::options_description generic("Generic options");
    generic.add_options()
            ("help,h", "produce help message");

    po::options_description config("Configuration");
    config.add_options()
            ("output,o", boost::program_options::value<std::string>()->default_value(""), "output directory")
            ("layer,l", boost::program_options::value<std::string>()->default_value(""), "output layer");

    po::options_description command("Hidden options");
    command.add_options()
            ("command", po::value<std::string>(), "command to execute, candidate: minus|to_csv")
            ("command_args", po::value<std::vector<std::string>>(), "command args");

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config).add(command);

    po::positional_options_description position;
    position.add("command", 1);
    position.add("command_args", -1);

    //process command line
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(cmdline_options).positional(position).run(), vm);
        po::notify(vm);
    } catch(std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch(...) {
        std::cerr << "Unknown error!\n";
        return 1;
    }

    //handle program options
    if (vm.count("help")) {
        std::cout << cmdline_options << "\n";
        return 0;
    }

    std::string command_string;
    if (vm.count("command")) {
        command_string = vm["command"].as<std::string>();
    }
    std::map<int, std::string> args_map;
    if (vm.count("command_args")) {
        std::vector<std::string> args = vm["command_args"].as<std::vector<std::string>>();
        for (int i = 0; i < args.size(); ++i) {
            args_map[i] = args[i];
        }
    }

    optional_args optional_args;
    if (vm.count("output")) {
        optional_args.output_path = {vm["output"].as<std::string>()};
    }
    if (vm.count("layer")) {
        optional_args.output_layer_name = {vm["layer"].as<std::string>()};
    }

    std::cout << "commands:" << command_string;
    for (const auto& [pos, arg] : args_map) {
        std::cout << " " << arg;
    }
    std::cout << std::endl;

    //execute command
    const auto [rc, message] = command_interpreter::instance()->execute(command_string, args_map, optional_args);
    if (rc != 0) {
        std::cout << "fail to execute command: " << command_string << ", message: " << message << std::endl;
    }
}