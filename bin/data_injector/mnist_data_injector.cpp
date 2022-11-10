#define CPU_ONLY

#include <chrono>
#include <glog/logging.h>

#include <network.hpp>
#include <configure_file.hpp>
#include <ml_layer.hpp>
#include <boost_serialization_wrapper.hpp>

#include "../env.hpp"

#include "data_generator.hpp"

configuration_file::json get_default_configuration()
{
	configuration_file::json output;
	output["dataset_path"] = "../../../dataset/MNIST/t10k-images.idx3-ubyte";
	output["dataset_label_path"] = "../../../dataset/MNIST/t10k-labels.idx1-ubyte";
	output["inject_interval_ms"] = 1000; //ms
	output["inject_amount"] = 8;
	output["ip_address"] = "127.0.0.1";
	output["ip_port"] = 8040;

    output["dataset_mode"] = "iid"; //default - randomly choose from dataset, iid - randomly choose from iid labels, non-iid - choose higher frequency labels for specific label
    configuration_file::json node_non_iid = configuration_file::json::object();
    for (int label_index = 0; label_index < 10; ++label_index) {
        node_non_iid[std::to_string(label_index)] = configuration_file::json::array({1.0,2.0});
    }
    output["non_iid_distribution"] = node_non_iid;

	return output;
}

enum class dataset_mode
{
    Default,
    IID,
    NonIID,
};

using DType = float;
constexpr int ml_dataset_all_possible_labels[] = {0,1,2,3,4,5,6,7,8,9};

int main(int argc, char **argv)
{
	std::string ip;
	uint16_t port;
	std::string dataset_path, label_dataset_path;
	std::chrono::milliseconds inject_interval;
	size_t inject_amount;
    dataset_mode dataset_label_mode;
    std::unordered_map<int, std::tuple<float, float>> dataset_label_distribution;

    //load configuration
    configuration_file config;
    config.SetDefaultConfiguration(get_default_configuration());
    auto return_code = config.LoadConfiguration(CONFIG_FILE_NAME::DFL_DATA_INJECTOR);
    if(return_code < configuration_file::NoError)
    {
        if (return_code == configuration_file::FileFormatError)
            LOG(FATAL) << "configuration file format error";
        else
            LOG(FATAL) << "configuration file error code: " << return_code;
    }

	//set dataset
	{
		auto dataset_path_opt = config.get<std::string>("dataset_path");
		if (dataset_path_opt) dataset_path = *dataset_path_opt;
		else LOG(FATAL) << "no dataset path provided";
		auto label_dataset_path_opt = config.get<std::string>("dataset_label_path");
		if (label_dataset_path_opt) label_dataset_path = *label_dataset_path_opt;
		else LOG(FATAL) << "no dataset label path provided";
        auto dataset_mode_opt = config.get<std::string>("dataset_mode");
        std::string dataset_mode_str;
        if (dataset_mode_opt) dataset_mode_str = *dataset_mode_opt;
        if (dataset_mode_str == "default")
            dataset_label_mode = dataset_mode::Default;
        else if (dataset_mode_str == "iid")
            dataset_label_mode = dataset_mode::IID;
        else if (dataset_mode_str == "non-iid")
        {
            dataset_label_mode = dataset_mode::NonIID;
            auto non_iid_distribution = config.get_json()["non_iid_distribution"];
            for (auto non_iid_item = non_iid_distribution.begin(); non_iid_item != non_iid_distribution.end(); ++non_iid_item)
            {
                int label = std::stoi(non_iid_item.key());
                auto min_max_array = *non_iid_item;
                float min = min_max_array.at(0);
                float max = min_max_array.at(1);
                if (max > min)
                {
                    dataset_label_distribution[label] = {min, max};
                }
                else
                {
                    dataset_label_distribution[label] = {max, min}; //swap the order
                }
            }
        }

        else
        {
            LOG(FATAL) << "no dataset label distribution provided";
        }
	}
	
	//set ip
	{
		auto ip_opt = config.get<std::string>("ip_address");
		if (ip_opt) ip = *ip_opt;
		else LOG(FATAL) << "no ip address in config file";
		auto port_opt = config.get<uint16_t>("ip_port");
		if (port_opt)
		{
			try
			{
				port = *port_opt;
			}
			catch (...)
			{
				LOG(FATAL) << "invalid port";
			}
		}
		else LOG(FATAL) << "no port in config file";
	}
	
	//set injection
	{
		auto inject_interval_opt = config.get<size_t>("inject_interval_ms");
		try
		{
			if (inject_interval_opt) inject_interval = std::chrono::milliseconds(*inject_interval_opt);
			else LOG(FATAL) << "no inject interval in config file";
		}
		catch (...)
		{
			LOG(FATAL) << "invalid inject interval";
		}
		auto inject_amount_opt = config.get<size_t>("inject_amount");
		try
		{
			if (inject_amount_opt) inject_amount = *inject_amount_opt;
			else LOG(FATAL) << "no inject amount in config file";
		}
		catch (...)
		{
			LOG(FATAL) << "invalid inject amount";
		}
	}

	//load dataset
	LOG(INFO) << "loading dataset";
	Ml::data_converter<DType> dataset;
	dataset.load_dataset_mnist(dataset_path, label_dataset_path);
	LOG(INFO) << "loading dataset - done";
	
	network::p2p client;
	bool _running = true;
	std::thread worker_thread([&](){
		auto loop_start_time = std::chrono::system_clock::now();
		int count = 0;
		while (_running)
		{
			count++;

            Ml::tensor_blob_like<DType> label;
            label.getShape() = {1};
            std::tuple<std::vector<Ml::tensor_blob_like<DType>>, std::vector<Ml::tensor_blob_like<DType>>> inject_data;
            switch (dataset_label_mode) {
                case dataset_mode::Default:
                {
                    inject_data = dataset.get_random_data(inject_amount);
                    break;
                }
                case dataset_mode::IID:
                {
                    std::vector<Ml::tensor_blob_like<DType>> train_data, train_label;
                    std::random_device dev;
                    std::mt19937 rng(dev());
                    std::uniform_int_distribution<int> distribution(0, 9);
                    for (int i = 0; i < inject_amount; ++i)
                    {
                        int label_int = ml_dataset_all_possible_labels[distribution(rng)];
                        label.getData() = {DType(label_int)};
                        auto[train_data_slice, train_label_slice] = dataset.get_random_data_by_Label(label, 1);
                        assert(!train_data_slice.empty() && !train_label_slice.empty());
                        train_data.insert(train_data.end(), train_data_slice.begin(), train_data_slice.end());
                        train_label.insert(train_label.end(), train_label_slice.begin(), train_label_slice.end());
                    }
                    inject_data = {train_data, train_label};
                    break;
                }
                case dataset_mode::NonIID:
                {
                    std::random_device dev;
                    std::mt19937 rng(dev());

                    Ml::non_iid_distribution<DType> label_distribution;
                    for (auto &target_label : ml_dataset_all_possible_labels)
                    {
                        label.getData() = {DType(target_label)};
                        auto iter = dataset_label_distribution.find(target_label);
                        if (iter != dataset_label_distribution.end())
                        {
                            auto[dis_min, dis_max] = iter->second;
                            if (dis_min == dis_max)
                            {
                                label_distribution.add_distribution(label, dis_min);
                            }
                            else
                            {
                                std::uniform_real_distribution<DType> distribution(dis_min, dis_max);
                                label_distribution.add_distribution(label, distribution(rng));
                            }
                        }
                        else
                        {
                            LOG(ERROR) << "cannot find the desired label";
                        }
                    }
                    inject_data = dataset.get_random_non_iid_dataset(label_distribution, inject_amount);
                    break;
                }
            }

            {
                auto inject_data_label = std::get<1>(inject_data);
                std::stringstream labels_ss;
                for (auto&& single_label: inject_data_label) {
                    labels_ss << "|" << single_label.getData()[0] << "| ";
                }
                LOG(INFO) << "Prepare data with labels: " << labels_ss.str();
            }

            std::stringstream inject_data_stream = serialize_wrap<boost::archive::binary_oarchive>(inject_data);
			std::string inject_data_str = inject_data_stream.str();
			client.send(ip,port,network::i_p2p_node::ipv4,inject_data_str.data(),inject_data_str.size(), [](network::p2p::send_packet_status status, const char* data, int length){
				std::string reply(data, length);
				if(status == network::p2p::send_packet_success)
				{
					LOG(INFO) << "Inject success";
				}
				else
				{
					LOG(INFO) << "Inject fail: " << network::i_p2p_node::send_packet_status_message[status];
				}
			});
			std::this_thread::sleep_until(loop_start_time + inject_interval * count);
		}
	});
	printf("press any key to stop\n");
	std::cin.get();
	printf("stopping ...\n");
	_running = false;
	worker_thread.join();
	return 0;
}

