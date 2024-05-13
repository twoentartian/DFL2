#pragma once

#include <string>
#include <random>
#include <configure_file.hpp>
#include <sstream>
#include <filesystem>
#include <optional>

#include "boost_serialization_wrapper.hpp"

#include <tmt.hpp>
#include <util.hpp>
#include <ml_layer/caffe_model_parameters.hpp>

#include "./node.hpp"
#include "./simulation_util.hpp"

#include "../tool/simulation_config_generator_common_functions.hpp"

enum class service_status
{
	success,
	fail_not_specified_reason,
	skipped
	
};

enum class service_trigger_type
{
    //process_per_tick trigger type
    start_of_tick,
    end_of_tick,
    start_of_training,
    end_of_training,
    start_of_averaging,
    end_of_averaging,

    //process_on_event trigger type
    model_before_training,
    model_after_training,
    model_after_averaging,
    model_added_to_average_buffer,

    service_trigger_type_index
};

template <typename model_datatype>
class service
{
public:
	bool enable;
	
	virtual std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) = 0;
	
	service()
	{
		node_vector_container = nullptr;
		node_container = nullptr;
		enable = false;
	}
	
	virtual std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>&, std::vector<node<model_datatype>*>&) = 0;
	
	virtual std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) = 0;

    virtual std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) = 0;
	
	virtual std::tuple<service_status, std::string> destruction_service() = 0;

protected:
	void set_node_container(std::map<std::string, node<model_datatype> *>& map, std::vector<node<model_datatype>*>& vector)
	{
		node_vector_container = &vector;
		node_container = &map;
	}
	
	std::vector<node<model_datatype>*>* node_vector_container;
	std::map<std::string, node<model_datatype> *>* node_container;
};

/// Notice that the accuracy_record service keeps its own caffe solver for testing the accuracy,
/// these solvers are independent from the solvers in simulator
template <typename model_datatype>
class accuracy_record : public service<model_datatype>
{
public:
    //set these variables before init
    std::string ml_solver_proto;
    Ml::data_converter<model_datatype>* test_dataset;
    int ml_test_batch_size;
private:
    std::optional<std::tuple<std::vector<const Ml::tensor_blob_like<model_datatype>*>, std::vector<const Ml::tensor_blob_like<model_datatype>*>>> fixed_test_dataset;
	int ml_test_interval_tick;
    bool use_fixed_test_dataset;
public:
	accuracy_record()
	{
		ml_test_batch_size = 0;
		ml_test_interval_tick = 0;
		ml_solver_proto = "";
		test_dataset = nullptr;
		solver_for_testing = nullptr;
	}
	
	std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
	{
		this->enable = config["enable"];
		this->ml_test_interval_tick = config["interval"];
        this->use_fixed_test_dataset = config["fixed_test_dataset"];
		
		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
	{
		this->set_node_container(_node_container, _node_vector_container);
		
		LOG_IF(FATAL, test_dataset == nullptr) << "test_dataset is not set";
		
		accuracy_file.reset(new std::ofstream(output_path / "accuracy.csv", std::ios::binary));
		*accuracy_file << "tick";
		for (auto &single_node : *(this->node_container))
		{
			*accuracy_file << "," << single_node.second->name;
		}
		*accuracy_file << std::endl;

        loss_file.reset(new std::ofstream(output_path / "loss.csv", std::ios::binary));
        *loss_file << "tick";
        for (auto &single_node : *(this->node_container))
        {
            *loss_file << "," << single_node.second->name;
        }
        *loss_file << std::endl;
		
		//solver for testing
		size_t solver_for_testing_size = std::thread::hardware_concurrency();
		solver_for_testing = new Ml::MlCaffeModel<float, caffe::SGDSolver>[solver_for_testing_size];
		for (int i = 0; i < solver_for_testing_size; ++i)
		{
			solver_for_testing[i].load_caffe_model(ml_solver_proto);
		}

        //pick fixed test dataset
        if (use_fixed_test_dataset) {
            std::vector<const Ml::tensor_blob_like<model_datatype>*> all_test_data, all_test_label;
            const std::vector<Ml::tensor_blob_like<model_datatype>>& all_labels = test_dataset->get_label();

            const std::unordered_map<std::string, std::vector<Ml::tensor_blob_like<model_datatype>>>& data_by_label = test_dataset->get_container_by_label();
            std::map<std::string, const Ml::tensor_blob_like<model_datatype>*> all_label_types;
            for (const auto& single_label : all_labels) {
                const std::string single_label_digest = single_label.get_str();
                if (!all_label_types.contains(single_label_digest)) all_label_types.emplace(single_label_digest, &single_label);
            }

            const int single_label_sample_size = ml_test_batch_size / all_label_types.size();
            LOG_IF(FATAL, single_label_sample_size*all_label_types.size() != ml_test_batch_size) << "test batch size % all_labels.size() != 0";
            for (const auto& [single_label_digest, single_label] : all_label_types) {
                const std::vector<Ml::tensor_blob_like<model_datatype>>& all_data_of_label = data_by_label.at(single_label_digest);
                for (int i = 0; i < single_label_sample_size; ++i) {
                    all_test_data.push_back(&all_data_of_label[i]);
                    all_test_label.push_back(single_label);
                }
            }
            fixed_test_dataset = {all_test_data, all_test_label};
        }

		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
	{
		if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::end_of_tick) return {service_status::skipped, "not service_trigger_type::end_of_tick"};

		if (tick % ml_test_interval_tick != 0) return {service_status::skipped, "not time yet"};

        tmt::ParallelExecution([&tick, this](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node) {
            std::vector<const Ml::tensor_blob_like<model_datatype>*> test_data, test_label;
            if (fixed_test_dataset.has_value()) {
                std::tie(test_data, test_label) = *fixed_test_dataset;
            }
            else {
                std::tie(test_data, test_label) = test_dataset->get_random_data(ml_test_batch_size);
            }
            auto model = single_node->solver->get_parameter();
            solver_for_testing[thread_index].set_parameter(model);
            model_datatype loss = 0;
            auto accuracy = solver_for_testing[thread_index].evaluation(test_data, test_label, &loss);
            single_node->nets_loss_only_record.emplace(tick, loss);
            single_node->nets_accuracy_only_record.emplace(tick, accuracy);
        }, this->node_vector_container->size(), this->node_vector_container->data());
        
        //print accuracy to file
        *accuracy_file << tick;
        for (auto &single_node: *(this->node_container))
        {
            auto iter_find = single_node.second->nets_accuracy_only_record.find(tick);
            if (iter_find != single_node.second->nets_accuracy_only_record.end())
            {
                auto accuracy = iter_find->second;
                *accuracy_file << "," << accuracy;
            }
            else
            {
                *accuracy_file << "," << " ";
            }
        }
        *accuracy_file << std::endl;

        //print loss to file
        *loss_file << tick;
        for (auto &single_node: *(this->node_container))
        {
            auto iter_find = single_node.second->nets_loss_only_record.find(tick);
            if (iter_find != single_node.second->nets_loss_only_record.end())
            {
                auto loss = iter_find->second;
                *loss_file << "," << loss;
            }
            else
            {
                *loss_file << "," << " ";
            }
        }
        *loss_file << std::endl;
        
		return {service_status::success, ""};
	}

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }
	
	std::tuple<service_status, std::string> destruction_service() override
	{
		delete[] solver_for_testing;
		accuracy_file->flush();
		accuracy_file->close();
        loss_file->flush();
        loss_file->close();
		
		return {service_status::success, ""};
	}
	
private:
	Ml::MlCaffeModel<float, caffe::SGDSolver>* solver_for_testing;
	std::shared_ptr<std::ofstream> accuracy_file;
    std::shared_ptr<std::ofstream> loss_file;
};

template <typename model_datatype>
class model_weights_difference_record : public service<model_datatype>
{
public:
	//set these variables before init
	int ml_model_weight_diff_record_interval_tick;
	
	model_weights_difference_record()
	{
		this->node_vector_container = nullptr;
		ml_model_weight_diff_record_interval_tick = 0;
	}
	
	std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
	{
		this->enable = config["enable"];
		this->ml_model_weight_diff_record_interval_tick = config["interval"];
		
		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
	{
		//LOG_IF(FATAL, node_vector_container == nullptr) << "node_vector_container is not set";
		this->set_node_container(_node_container, _node_vector_container);
		
		model_weights_file.reset(new std::ofstream(output_path / "model_weight_diff.csv", std::ios::binary));
        LOG_ASSERT(model_weights_file->good());
		*model_weights_file << "tick";
        
        const auto& weights = (*this->node_vector_container)[0]->solver->get_parameter();
		const auto& layers = weights.getLayers();
		
		for (auto& single_layer: layers)
		{
			*model_weights_file << "," << single_layer.getName();
		}
		*model_weights_file << std::endl;
		
		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
	{
		if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::end_of_tick) return {service_status::skipped, "not service_trigger_type::end_of_tick"};

        if (tick % ml_model_weight_diff_record_interval_tick != 0) return {service_status::skipped, "not time yet"};
        
        const auto& weights = (*this->node_vector_container)[0]->solver->get_parameter();
        const auto& layers = weights.getLayers();
        size_t number_of_layers = layers.size();
        auto *weight_diff_sums = new std::atomic<float>[number_of_layers];
        for (int i = 0; i < number_of_layers; ++i) weight_diff_sums[i] = 0;
        
        tmt::ParallelExecution([this, &weight_diff_sums, &number_of_layers](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node)
                               {
                                   uint32_t index_next = index + 1;
                                   const uint32_t total_size = this->node_vector_container->size();
                                   if (index_next == total_size - 1) index_next = 0;
                                   auto net1 = (*this->node_vector_container)[index]->solver->get_parameter();
                                   auto net2 = (*this->node_vector_container)[index_next]->solver->get_parameter();
                                   auto layers1 = net1.getLayers();
                                   auto layers2 = net2.getLayers();
                                   for (int i = 0; i < number_of_layers; ++i)
                                   {
                                       auto diff = layers1[i] - layers2[i];
                                       diff.abs();
                                       auto value = diff.sum();
                                       weight_diff_sums[i] = weight_diff_sums[i] + value;
                                   }
                               }, this->node_vector_container->size() - 1, this->node_vector_container->data());
        
        *model_weights_file << tick;
        for (int i = 0; i < number_of_layers; ++i)
        {
            *model_weights_file << "," << weight_diff_sums[i];
        }
        *model_weights_file << std::endl;
        delete[] weight_diff_sums;
        
		return {service_status::success, ""};
	}

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }
	
	std::tuple<service_status, std::string> destruction_service() override
	{
		model_weights_file->flush();
		model_weights_file->close();
		
		return {service_status::success, ""};
	}

private:
	std::shared_ptr<std::ofstream> model_weights_file;
};

template <typename model_datatype>
class model_abs_change_during_averaging : public service<model_datatype>
{
public:
    model_abs_change_during_averaging()
    {
        this->node_vector_container = nullptr;
    }

    std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
    {
        this->enable = config["enable"];

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
    {
        //LOG_IF(FATAL, node_vector_container == nullptr) << "node_vector_container is not set";
        this->set_node_container(_node_container, _node_vector_container);

        if (this->enable == false) return {service_status::skipped, "not enabled"};

        model_abs_change_file.reset(new std::ofstream(output_path / "model_abs_change_per_averaging.txt", std::ios::binary));
        LOG_ASSERT(model_abs_change_file->good());

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        //nothing to do
        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};
        if (trigger == service_trigger_type::model_added_to_average_buffer) {
            const auto& parameter = *((*this->node_container)[triggered_node_name]->simulation_service_data.just_received_model_ptr);
            auto abs = get_parameter_abs(parameter);
            abs_of_received_models[triggered_node_name].push_back(abs);
        }
        if (trigger == service_trigger_type::model_after_averaging) {
            const auto& parameter = *((*this->node_container)[triggered_node_name]->simulation_service_data.average_output_ptr);
            const auto iter_abs_of_received_models_for_triggered_node = abs_of_received_models.find(triggered_node_name);
            LOG_ASSERT(iter_abs_of_received_models_for_triggered_node != abs_of_received_models.end());
            {
                std::lock_guard guard(model_abs_change_file_lock);
                *model_abs_change_file << "averaging at tick:" << tick << ",node:" << triggered_node_name <<",";
                for (const auto& v : iter_abs_of_received_models_for_triggered_node->second) {
                    *model_abs_change_file << v << " ";
                }
                iter_abs_of_received_models_for_triggered_node->second.clear();
                *model_abs_change_file << " -->> " << get_parameter_abs(parameter) << std::endl;
            }
        }

        if (trigger == service_trigger_type::model_before_training) {
            const auto& parameter = *((*this->node_container)[triggered_node_name]->simulation_service_data.model_before_training_ptr);
            abs_of_model_before_training[triggered_node_name] = get_parameter_abs(parameter);
        }
        if (trigger == service_trigger_type::model_after_training) {
            const auto& parameter = *((*this->node_container)[triggered_node_name]->simulation_service_data.model_before_training_ptr);
            const auto iter_abs_of_model_before_training_for_triggered_node = abs_of_model_before_training.find(triggered_node_name);
            LOG_ASSERT(iter_abs_of_model_before_training_for_triggered_node != abs_of_model_before_training.end());
            {
                std::lock_guard guard(model_abs_change_file_lock);
                *model_abs_change_file << "training at tick:" << tick << ",node:" << triggered_node_name <<",";
                *model_abs_change_file << iter_abs_of_model_before_training_for_triggered_node->second << " ";
                *model_abs_change_file << " -->> " << get_parameter_abs(parameter) << std::endl;
            }
        }

        return {service_status::fail_not_specified_reason, "not implemented"};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {
        if (model_abs_change_file) {
            model_abs_change_file->flush();
            model_abs_change_file->close();
        }

        return {service_status::success, ""};
    }

private:
    std::shared_ptr<std::ofstream> model_abs_change_file;
    std::mutex model_abs_change_file_lock;
    std::map<std::string, std::vector<model_datatype>> abs_of_received_models;
    std::map<std::string, model_datatype> abs_of_model_before_training;

    model_datatype get_parameter_abs(const Ml::caffe_parameter_net<model_datatype>& parameter) {
        model_datatype total = 0;
        for (const Ml::caffe_parameter_layer<model_datatype>& layer : parameter.getLayers()) {
            for (const auto& blob_vec : layer.getBlob_p()) {
                const Ml::tensor_blob_like<model_datatype>& tensor_blob = *blob_vec;
                for (const auto& v : tensor_blob.getData()) {
                    total += v*v;
                }
            }
        }
        return std::sqrt(total);
    }
};

template <typename model_datatype>
class model_weights_variance_record : public service<model_datatype>
{
public:
    //set these variables before init
    int model_weights_variance_record_interval_tick;
    
    model_weights_variance_record()
    {
        this->node_vector_container = nullptr;
        model_weights_variance_record_interval_tick = 0;
    }
    
    std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
    {
        this->enable = config["enable"];
        this->model_weights_variance_record_interval_tick = config["interval"];
        
        return {service_status::success, ""};
    }
    
    std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
    {
        this->set_node_container(_node_container, _node_vector_container);

        if (this->enable == false) return {service_status::skipped, "not enabled"};
        
        model_weights_file.reset(new std::ofstream(output_path / "model_weight_var.csv", std::ios::binary));
        LOG_ASSERT(model_weights_file->good());
        *model_weights_file << "tick";
    
        bool first_node = true;
        for (const auto& [node_name, node] : *(this->node_container)) {
            const Ml::caffe_parameter_net<model_datatype>& model = node->solver->get_parameter();
            const std::vector<Ml::caffe_parameter_layer<model_datatype>>& layers = model.getLayers();
    
            for (auto& single_layer: layers)
            {
                if (single_layer.getBlob_p().size() == 0) continue;
                *model_weights_file << "," << node_name << "-" << single_layer.getName();
                if (first_node) {
                    layer_order.push_back(single_layer.getName());
                }
            }
            first_node = false;
            node_order.push_back(node_name);
        }
        *model_weights_file << std::endl;
        
        return {service_status::success, ""};
    }
    
    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (tick % model_weights_variance_record_interval_tick != 0) return {service_status::skipped, "not time yet"};

        float tick_float = tick;
        bool triggered = false;
        if (trigger == service_trigger_type::start_of_tick) {
            tick_float = tick_float;
            triggered = true;
        }
        if (trigger == service_trigger_type::end_of_training) {
            tick_float = tick_float + 0.5;
            triggered = true;
        }
        if (trigger == service_trigger_type::end_of_averaging) {
            tick_float = tick_float + 0.9;
            triggered = true;
        }

        if (triggered) {
            std::map<std::string, std::map<std::string, model_datatype>> variances;
            std::mutex variances_lock;

            tmt::ParallelExecution([this, &variances, &variances_lock](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node)
                                   {
                                       std::map<std::string, model_datatype> vars;
                                       const auto& model = single_node->solver->get_parameter();
                                       const std::vector<Ml::caffe_parameter_layer<model_datatype>>& layers = model.getLayers();
                                       for (const Ml::caffe_parameter_layer<model_datatype>& single_layer : layers)
                                       {
                                           const auto& blobs = single_layer.getBlob_p();
                                           if (blobs.size() == 0) continue;
                                           model_datatype var = calculate_variance(blobs[0]->getData()); //we only care about the normal weights, not biased weights.
                                           vars.emplace(single_layer.getName(), var);
                                       }
                                       {
                                           std::lock_guard guard(variances_lock);
                                           variances[single_node->name] = vars;
                                       }
                                   }, this->node_vector_container->size(), this->node_vector_container->data());

            *model_weights_file << tick_float;
            for (const auto& node_name : this->node_order)
            {
                const auto vars = variances.at(node_name);
                for (const auto& layer_name : this->layer_order)
                {
                    const auto variance = vars.at(layer_name);
                    *model_weights_file << "," << variance;
                }
            }
            *model_weights_file << std::endl;
        }

        return {service_status::success, ""};
    }
    
    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }
    
    std::tuple<service_status, std::string> destruction_service() override
    {
        model_weights_file->flush();
        model_weights_file->close();
        
        return {service_status::success, ""};
    }

private:
    std::shared_ptr<std::ofstream> model_weights_file;
    std::vector<std::string> layer_order;
    std::vector<std::string> node_order;
    
    model_datatype calculate_mean(const std::vector<model_datatype>& data) {
        model_datatype sum = 0.0;
        for (const double& value : data) {
            sum += value;
        }
        return sum / static_cast<double>(data.size());
    }
    
    model_datatype calculate_variance(const std::vector<model_datatype>& data) {
        if (data.empty()) {
            return 0.0; // Handle empty data
        }
    
        model_datatype mean = calculate_mean(data);
        model_datatype variance = 0.0;
        
        for (const double& value : data) {
            double diff = value - mean;
            variance += diff * diff;
        }
        
        return variance / static_cast<model_datatype>(data.size());
    }
};

template <typename model_datatype>
class force_broadcast_model : public service<model_datatype>
{
private:
    service_trigger_type to_ConfigItem_type(const std::string& str) {
        if (str == "before_train") {
            return service_trigger_type::start_of_training;
        }
        else if (str == "after_train") {
            return service_trigger_type::end_of_training;
        }
        else if (str == "before_average") {
            return service_trigger_type::start_of_averaging;
        }
        else if (str == "after_average") {
            return service_trigger_type::end_of_averaging;
        }
        else {
            LOG(FATAL) << "unknown ConfigItem_type: " << str;
        }
        return service_trigger_type::service_trigger_type_index;
    }

    enum ConfigItem_trigger_time {
        at,
        every,

        ConfigItem_trigger_type_Last
    };

    ConfigItem_trigger_time to_ConfigItem_trigger_time(const std::string& str) {
        if (str == "at") {
            return ConfigItem_trigger_time::at;
        }
        else if (str == "every") {
            return ConfigItem_trigger_time::every;
        }
        else {
            LOG(FATAL) << "unknown ConfigItem_trigger_type: " << str;
            return ConfigItem_trigger_time::ConfigItem_trigger_type_Last;
        }
    }

    struct ConfigItem {
        service_trigger_type type;
        ConfigItem_trigger_time at_or_every;
        std::set<int> ticks;
        std::optional<std::string> src_node_name;
    };

    ConfigItem parseConfigItem(const std::string &item) {
        ConfigItem result;
        auto colonPositions = util::split(item, ':');  // Use the split function to find all colon positions.

        if (colonPositions.size() < 3) {
            LOG(FATAL) << "Invalid input format for item: " << item;
            return result;
        }

        // Extract category, type, and the rest
        result.type = to_ConfigItem_type(colonPositions[0]);
        result.at_or_every = to_ConfigItem_trigger_time(colonPositions[1]);

        // The last part could be the optional item or part of the number sequence
        std::string numbersPart = colonPositions[2];
        if (colonPositions.size() > 3) {
            // If more than 3 parts, the last part is the optional item
            result.src_node_name = {colonPositions[3]};
        }

        // Split the numbersPart by ',' to extract numbers
        auto numberTokens = util::split(numbersPart, ',');
        for (const auto &token : numberTokens) {
            if (!token.empty()) {
                result.ticks.emplace(std::stoi(token));
            }
        }

        return result;
    }

private:
    std::map<service_trigger_type, std::vector<ConfigItem>> all_config_items;

public:
	force_broadcast_model()
	{

	}
	
	std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
	{
		this->enable = config["enable"];
		std::vector<std::string> all_config_items_str = config["config_items"];
        for (const auto& config_str : all_config_items_str) {
            ConfigItem configItem = parseConfigItem(config_str);
            all_config_items[configItem.type].push_back(configItem);
        }
		
		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
	{
		this->set_node_container(_node_container, _node_vector_container);
		
		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
	{
		if (this->enable == false) return {service_status::skipped, "not enabled"};

        bool triggered = false;
        const auto triggered_type = all_config_items.find(trigger);
        const ConfigItem* triggered_config_item;
        if (triggered_type != all_config_items.end()) { //get something to do
            const auto& triggered_items = triggered_type->second;
            for (const ConfigItem& each_item : triggered_items) {
                if (each_item.at_or_every == ConfigItem_trigger_time::at) {
                    if (each_item.ticks.contains(tick)) {
                        triggered = true;
                        triggered_config_item = &each_item;
                        break;
                    }
                }
                else if (each_item.at_or_every == ConfigItem_trigger_time::every) {
                    if (tick % *each_item.ticks.begin() == 0 and tick != 0) {
                        triggered = true;
                        triggered_config_item = &each_item;
                        break;
                    }
                }
                else {
                    LOG(FATAL) << "unknown ConfigItem_trigger_time";
                }
            }
        }
        if (triggered) {
            trigger_forced_broadcast(triggered_config_item->src_node_name, tick);
        }
		return {service_status::success, ""};
	}

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }
	
	std::tuple<service_status, std::string> destruction_service() override
	{
		return {service_status::success, ""};
	}

private:
    void trigger_forced_broadcast(std::optional<std::string> src_node_name, int tick) {
        if (src_node_name.has_value()) {
            LOG(INFO) << "force_broadcast_model, src node:" << *src_node_name << ", triggered at tick: " << tick;
            auto node_iter = this->node_container->find(*src_node_name);
            LOG_IF(FATAL, node_iter == this->node_container->end()) << *src_node_name << " does not exist";
            const auto& model = node_iter->second->solver->get_parameter();
            for (auto& node: *(this->node_container)) {
                node.second->solver->set_parameter(model);
            }
        }
        else {
            LOG(INFO) << "force_broadcast_model triggered at tick: " << tick;
            auto model_sum = (*this->node_vector_container)[0]->solver->get_parameter();
            model_sum.set_all(0);

            for (auto& node: *(this->node_container))
            {
                model_sum = model_sum + node.second->solver->get_parameter();
            }
            model_sum = model_sum / this->node_container->size();

            for (auto& node: *(this->node_container))
            {
                node.second->solver->set_parameter(model_sum);
            }
        }
    }
};

template <typename model_datatype>
class time_based_hierarchy_service : public service<model_datatype>
{
public:
	enum fedavg_buffer_size_strategy
	{
		unknown_strategy = 0,
		static_strategy,
		linear_strategy
	};
	
public:
	std::unordered_map<std::string, int> last_time_changed;
	std::unordered_map<std::string, int> as_peer_count;
	fedavg_buffer_size_strategy fedavg_buffer_strategy;
	int least_peer_change_interval;
	float accuracy_threshold_high;
	float accuracy_threshold_low;
	std::string ml_solver_proto;
	Ml::data_converter<model_datatype>* test_dataset;
	int ml_test_batch_size;
	std::vector<int>* ml_dataset_all_possible_labels;
	
	std::shared_ptr<std::ofstream> peer_change_file;
	
	time_based_hierarchy_service()
	{
		fedavg_buffer_strategy = unknown_strategy;
		least_peer_change_interval = 0;
		accuracy_threshold_high = 0.0f;
		accuracy_threshold_low = 0.0f;
		
		size_t solver_for_testing_size = std::thread::hardware_concurrency();
		solver_for_testing = new Ml::MlCaffeModel<model_datatype, caffe::SGDSolver>[solver_for_testing_size];
	}
	
	std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
	{
		this->enable = config["enable"];
		this->least_peer_change_interval = config["least_peer_change_interval"];
		accuracy_threshold_high = config["accuracy_threshold_high"];
		accuracy_threshold_low = config["accuracy_threshold_low"];
		LOG_IF(FATAL, accuracy_threshold_low >= accuracy_threshold_high) << "accuracy_threshold_low < accuracy_threshold_high must satisfy";
		
		std::string fedavg_buffer_strategy_str = config["fedavg_buffer_size"];
		if (fedavg_buffer_strategy_str == "static") fedavg_buffer_strategy = fedavg_buffer_size_strategy::static_strategy;
		else if (fedavg_buffer_strategy_str == "linear") fedavg_buffer_strategy = fedavg_buffer_size_strategy::linear_strategy;
		else LOG(FATAL) << "unknown fedavg_buffer_strategy in node";
		
		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
	{
		this->set_node_container(_node_container, _node_vector_container);
		
		if (this->enable == false)
		{
			//copy planned peers to peers
			for (auto& node: *(this->node_container))
			{
				node.second->peers = node.second->planned_peers;
			}
		}
		else
		{
			//do nothing, add peers in the future
			
			//solver for testing
			size_t solver_for_testing_size = std::thread::hardware_concurrency();
			for (int i = 0; i < solver_for_testing_size; ++i)
			{
				solver_for_testing[i].load_caffe_model(ml_solver_proto);
			}
			
			peer_change_file.reset(new std::ofstream(output_path / "peer_change_record.txt", std::ios::binary));
		}
		
		return {service_status::success, ""};
	}
	
	std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
	{
		if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::end_of_tick) return {service_status::skipped, "not service_trigger_type::end_of_tick"};
		
		//calculate accuracy
		tmt::ParallelExecution_StepIncremental([&tick, this](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node)
		                       {
			                       if (tick - single_node->last_measured_tick < least_peer_change_interval) return;
			                       auto[test_data, test_label] = get_dataset_by_node_type(*test_dataset, *single_node, ml_test_batch_size, *ml_dataset_all_possible_labels);
                                   auto model = single_node->solver->get_parameter();
			                       solver_for_testing[thread_index].set_parameter(model);
			                       auto accuracy = solver_for_testing[thread_index].evaluation(test_data, test_label);
			                       single_node->nets_accuracy_only_record.emplace(tick, accuracy);
			                       single_node->last_measured_accuracy = accuracy;
			                       single_node->last_measured_tick = tick;
		                       }, this->node_vector_container->size(), this->node_vector_container->data());
		
		//process peers
		for (auto node_pair: *(this->node_container))
		{
			node<model_datatype>* node_pointer = node_pair.second;
			
			//do nothing if it has changed peers recently
			if (tick - last_time_changed[node_pointer->name] < least_peer_change_interval) continue;
			
			if (node_pointer->last_measured_accuracy >= accuracy_threshold_high && node_pointer->peers.size() != node_pointer->planned_peers.size())
			{
				//try add a peer
				bool add = false;
				std::string new_peer_name;
				for (const auto& [name, single_planned_peer]: node_pointer->planned_peers)
				{
					if (node_pointer->peers.find(name) == node_pointer->peers.end())
					{
						node_pointer->peers.emplace(name, single_planned_peer);
						new_peer_name = name;

						add = true;
						break;
					}
				}
				LOG_IF(FATAL, add == false) << "DFL simulator logic error, no node added to the peers";
				
				//update the buffer size for new peer
				as_peer_count[new_peer_name]++;
				auto& new_peer = *(this->node_container->at(new_peer_name));
				new_peer.buffer_size = std::round((float (new_peer.planned_buffer_size) / new_peer.planned_peers.size()) * as_peer_count[new_peer_name]);
				
				std::stringstream ss;
				ss << "tick:" << tick << "    " << node_pointer->name << "(accuracy: " << node_pointer->last_measured_accuracy << ") add " << new_peer_name << "(buffer size:" << new_peer.buffer_size << ")";
				*peer_change_file << ss.str() << std::endl;
				LOG(INFO) << "[time_based_hierarchy_service]  " << ss.str();
				last_time_changed[node_pointer->name] = tick;
			}
			
			if (node_pointer->last_measured_accuracy <= accuracy_threshold_low && node_pointer->peers.size() != 0)
			{
				//try delete a peer
				static std::random_device rd;
				std::uniform_int_distribution<int> distribution(0, node_pointer->peers.size()-1);
				int delete_index = distribution(rd);
				std::string delete_peer_name;
				for (const auto& [name, single_peer]: node_pointer->peers)
				{
					if (delete_index == 0)
					{
						delete_peer_name = name;
						as_peer_count[delete_peer_name]--;
						auto& delete_peer = *(this->node_container->at(delete_peer_name));
						delete_peer.buffer_size = std::round((float (delete_peer.planned_buffer_size) / delete_peer.planned_peers.size()) * as_peer_count[delete_peer_name]);
						std::stringstream ss;
						ss << "tick:" << tick << "    " << node_pointer->name << "(accuracy: " << node_pointer->last_measured_accuracy << ") delete " << delete_peer_name << "(buffer size:" << delete_peer.buffer_size << ")";
						*peer_change_file << ss.str() << std::endl;
						LOG(INFO) << "[time_based_hierarchy_service]  " << ss.str();
						break;
					}
					delete_index --;
				}
				node_pointer->peers.erase(delete_peer_name);
				last_time_changed[node_pointer->name] = tick;
			}
			
		}
		
		return {service_status::success, ""};
	}

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }
	
	std::tuple<service_status, std::string> destruction_service() override
	{
		if (peer_change_file && peer_change_file->is_open())
		{
			peer_change_file->flush();
			peer_change_file->close();
		}
        delete[] solver_for_testing;
		
		return {service_status::success, ""};
	}

private:
	Ml::MlCaffeModel<float, caffe::SGDSolver>* solver_for_testing;
};

template <typename model_datatype>
class reputation_record : public service<model_datatype>
{

private:
    std::unordered_map<std::string, std::shared_ptr<std::ofstream>> reputations_files;
    
public:
    reputation_record()
    {
    
    }
    
    std::tuple<service_status, std::string> apply_config(const configuration_file::json &config) override
    {
        this->enable = config["enable"];
        
        return {service_status::success, ""};
    }
    
    std::tuple<service_status, std::string> init_service(const std::filesystem::path &output_path, std::map<std::string, node<model_datatype> *> &_node_container, std::vector<node<model_datatype> *> &_node_vector_container) override
    {
        this->set_node_container(_node_container, _node_vector_container);
        
        if (!this->enable) return {service_status::skipped, "not enabled"};
        
        //reputation folder
        std::filesystem::path reputation_folder = output_path / "reputation";
        if (!std::filesystem::exists(reputation_folder)) std::filesystem::create_directories(reputation_folder);
    
        for (auto& [node_name, node] : _node_container)
        {
            auto reputation_file = std::make_shared<std::ofstream>();
            reputation_file->open(reputation_folder / (node_name + "_reputation.csv"));
            reputations_files.emplace(node_name, reputation_file);
        }
    
        //print reputation first line
        for (auto &[node_name, node]: _node_container)
        {
            auto reputation_file = reputations_files[node_name];
            *reputation_file << "tick";
            for (auto &reputation_item: node->reputation_map)
            {
                *reputation_file << "," << reputation_item.first;
            }
            *reputation_file << std::endl;
        }
    
        return {service_status::success, ""};
    }
    
    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (!this->enable) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::end_of_tick) return {service_status::skipped, "not service_trigger_type::end_of_tick"};
        
        //print reputation map
        for (auto& [node_name, node]: *(this->node_container))
        {
            if (!node->model_averaged) continue; // we don't update reputation if node doesn't perform Fedavg
            
            auto reputation_file = reputations_files[node_name];
            *reputation_file << tick;
            for (auto &[target_reputation_node_name, reputation_value]: node->reputation_map)
            {
                *reputation_file << "," << reputation_value;
            }
            *reputation_file << std::endl;
        }
        
        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }
    
    std::tuple<service_status, std::string> destruction_service() override
    {
        if (!this->enable) return {service_status::skipped, "not enabled"};
        
        for (auto& [node_name, reputation_file]: reputations_files)
        {
            if (reputation_file)
            {
                reputation_file->flush();
                reputation_file->close();
            }
        }
        return {service_status::success, ""};
    }
    
};

template <typename model_datatype>
class model_record : public service<model_datatype>
{
private:
    int ml_model_record_interval_tick;
    std::string path;
    std::filesystem::path storage_path;
    std::set<std::string> nodes_to_record;
    bool final_record;
    
public:
    int total_tick;
    
public:
    model_record()
    {
        this->node_vector_container = nullptr;
        ml_model_record_interval_tick = 0;
    }
    
    std::tuple<service_status, std::string> apply_config(const configuration_file::json &config) override
    {
        this->enable = config["enable"];
        this->ml_model_record_interval_tick = config["interval"];
        this->path = config["path"];
        this->final_record = config["final_record"];
        const std::string nodes_to_record_str = config["nodes"];
        for (const auto& node_name : util::split(nodes_to_record_str, ','))
        {
            this->nodes_to_record.emplace(node_name);
        }
        
        return {service_status::success, ""};
    }
    
    std::tuple<service_status, std::string> init_service(const std::filesystem::path &output_path, std::map<std::string, node<model_datatype> *> &_node_container, std::vector<node<model_datatype> *> &_node_vector_container) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        this->set_node_container(_node_container, _node_vector_container);
        this->storage_path = output_path / path;
        if (!std::filesystem::exists(storage_path)) std::filesystem::create_directories(storage_path);
        
        //check available space
        const std::filesystem::space_info si = std::filesystem::space(this->storage_path);
        auto model = _node_vector_container[0]->solver->get_parameter();
        const std::uintmax_t model_size = serialize_wrap<boost::archive::binary_oarchive>(model).str().size();
        const std::uintmax_t space_required = model_size * (total_tick / ml_model_record_interval_tick + 1) * this->node_vector_container->size();
        if (si.available < space_required)
            LOG(FATAL) << "[model record service] not enough space in model record path, available: " << si.available/1024/1024 << "MB, required: " << space_required/1024/1024 << "MB";
        LOG(INFO) << "[model record service] space in model record path, available: " << si.available / 1024 / 1024 << "MB, required: " << space_required / 1024 / 1024 << "MB";
        return {service_status::success, ""};
    }
    
    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::end_of_tick) return {service_status::skipped, "not service_trigger_type::end_of_tick"};
        if (tick % ml_model_record_interval_tick == 0) {
            tmt::ParallelExecution([&tick, this](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node)
                                   {
                                       if (!this->nodes_to_record.contains(single_node->name)) return;
                                       std::filesystem::path folder_of_this_node = storage_path / single_node->name;
                                       if (!std::filesystem::exists(folder_of_this_node)) std::filesystem::create_directories(folder_of_this_node);
                                       auto model = single_node->solver->get_parameter();
                                       std::ofstream output_file(folder_of_this_node / (std::to_string(tick) + ".bin"));
                                       output_file << serialize_wrap<boost::archive::binary_oarchive>(model).str();
                                       output_file.close();
                                   }, this->node_vector_container->size(), this->node_vector_container->data());
        }
        if (tick == total_tick) {
            tmt::ParallelExecution([&tick, this](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node)
                                   {
                                       if (!this->nodes_to_record.contains(single_node->name)) return;
                                       std::filesystem::path folder = storage_path / "final";
                                       if (!std::filesystem::exists(folder)) std::filesystem::create_directories(folder);
                                       auto model = single_node->solver->get_parameter();
                                       std::ofstream output_file(folder / (single_node->name + ".bin"));
                                       output_file << serialize_wrap<boost::archive::binary_oarchive>(model).str();
                                       output_file.close();
                                   }, this->node_vector_container->size(), this->node_vector_container->data());
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }
    
    std::tuple<service_status, std::string> destruction_service() override
    {
        return {service_status::success, ""};
    }
};

template <typename model_datatype>
class received_model_record : public service<model_datatype>
{
private:
    std::string path;
    std::filesystem::path storage_path;
    std::set<std::string> nodes_to_record;

public:

public:
    received_model_record()
    {
        this->node_vector_container = nullptr;
    }

    std::tuple<service_status, std::string> apply_config(const configuration_file::json &config) override
    {
        this->enable = config["enable"];
        this->path = config["path"];
        const std::string nodes_to_record_str = config["nodes"];
        for (const auto& node_name : util::split(nodes_to_record_str, ','))
        {
            this->nodes_to_record.emplace(node_name);
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path &output_path, std::map<std::string, node<model_datatype> *> &_node_container, std::vector<node<model_datatype> *> &_node_vector_container) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        this->set_node_container(_node_container, _node_vector_container);
        this->storage_path = output_path / path;
        if (!std::filesystem::exists(storage_path)) std::filesystem::create_directories(storage_path);

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        return {service_status::skipped, "I only work on process_on_event"};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (!this->nodes_to_record.contains(triggered_node_name)) return {service_status::skipped, "node does not record"};

        if (trigger != service_trigger_type::model_added_to_average_buffer) return {service_status::skipped, "not service_trigger_type::model_received"};

        const auto node_iter = this->node_container->find(triggered_node_name);
        LOG_IF(FATAL, node_iter == this->node_container->end()) << triggered_node_name << " does not exist";

        std::filesystem::path folder_of_this_node = storage_path / triggered_node_name;
        if (!std::filesystem::exists(folder_of_this_node)) std::filesystem::create_directories(folder_of_this_node);
        std::filesystem::path folder_of_this_node_tick = folder_of_this_node / std::to_string(tick);
        if (!std::filesystem::exists(folder_of_this_node_tick)) std::filesystem::create_directories(folder_of_this_node_tick);

        const std::string& model_source_node_name = node_iter->second->simulation_service_data.just_received_model_source_node_name;
        const auto* model = node_iter->second->simulation_service_data.just_received_model_ptr;
        std::ofstream output_file(folder_of_this_node_tick / (model_source_node_name + ".bin"), std::ios::binary);
        LOG_IF(WARNING, !output_file.good()) << "cannot open file:" << folder_of_this_node_tick / (model_source_node_name + ".bin");
        output_file << serialize_wrap<boost::archive::binary_oarchive>(*model).str();
        output_file.close();

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {
        return {service_status::success, ""};
    }
};

template <typename model_datatype>
class apply_received_model : public service<model_datatype>
{
private:
    std::string path;
    std::filesystem::path storage_path;
    std::map<std::string, std::map<int, std::map<std::string, std::filesystem::path>>> all_received_models;

public:
    std::function<void(const std::string&, const Ml::caffe_parameter_net<model_datatype>&)> send_model_to_update_buffer_handler;

public:
    apply_received_model()
    {
        this->node_vector_container = nullptr;
    }

    std::tuple<service_status, std::string> apply_config(const configuration_file::json &config) override
    {
        this->enable = config["enable"];
        this->path = config["path"];
        this->storage_path = std::filesystem::path(this->path);

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path &output_path, std::map<std::string, node<model_datatype> *> &_node_container, std::vector<node<model_datatype> *> &_node_vector_container) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        this->set_node_container(_node_container, _node_vector_container);
        LOG_IF(FATAL, !std::filesystem::exists(storage_path)) << storage_path << " does not exist";

        for (const auto& entry_0 : std::filesystem::directory_iterator(this->storage_path)) {
            LOG_IF(FATAL, !std::filesystem::is_directory(entry_0)) << entry_0 << " is not a directory";
            //get the node names
            const auto& node_path = entry_0.path();
            std::map<int, std::map<std::string, std::filesystem::path>> all_ticks;
            for (const auto& entry_1 : std::filesystem::directory_iterator(node_path)) {
                LOG_IF(FATAL, !std::filesystem::is_directory(entry_1)) << entry_1 << " is not a directory";
                //get the tick
                const auto& tick_path = entry_1.path();
                std::map<std::string, std::filesystem::path> all_src_nodes_in_tick;
                for (const auto& entry_2 : std::filesystem::directory_iterator(tick_path)) {
                    const auto& src_node_path = entry_2.path();
                    const auto& src_node_name = src_node_path.filename();
                    all_src_nodes_in_tick[src_node_name] = src_node_path;
                }
                int tick = std::stoi(tick_path.filename());
                all_ticks[tick] = all_src_nodes_in_tick;
            }
            const std::string node_name = node_path.filename();
            all_received_models[node_name] = all_ticks;
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::start_of_averaging) return {service_status::skipped, "not service_trigger_type::start_of_averaging"};

        for (const auto& [node_name, all_ticks] : all_received_models) {
            const auto tick_iter = all_ticks.find(tick);
            if (tick_iter == all_ticks.end()) continue; //nothing to do for this tick

            const auto node_iter = this->node_container->find(node_name);
            LOG_IF(FATAL, node_iter==this->node_container->end()) << node_name << " not found";
            //find node
            for (const auto& [src_node_name, model_path] : tick_iter->second) {
                std::ifstream input_file(model_path, std::ios::binary);
                LOG_IF(WARNING, !input_file.good()) << "cannot open file:" << model_path.string();
                std::stringstream ss;
                ss << input_file.rdbuf();
                input_file.close();
                const Ml::caffe_parameter_net<model_datatype> model = deserialize_wrap<boost::archive::binary_iarchive, Ml::caffe_parameter_net<model_datatype>>(ss);
                send_model_to_update_buffer_handler(node_name, model);
            }
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {
        return {service_status::success, ""};
    }
};

template <typename model_datatype>
class network_topology_manager : public service<model_datatype>
{
private:
    bool enable_read_from_file;
    std::string read_from_file_path;

    bool enable_scale_free_network;
    float scale_free_network_gamma;
    int scale_free_network_min_peer;
    float scale_free_network_buffer_to_peer_ratio;
    int scale_free_network_interval;

    bool enable_connection_pair_swap_config;
    float connection_pair_swap_percentage;
    int connection_pair_swap_interval;

    std::filesystem::path output_file_path;
    std::ofstream output_file;
    int last_topology_update_tick;
public:


public:
    network_topology_manager()
    {
        this->node_vector_container = nullptr;

    }

    std::tuple<service_status, std::string> apply_config(const configuration_file::json &config) override
    {
        {
            auto read_from_file_config = config["read_from_file"];
            enable_read_from_file = read_from_file_config["enable"];
            read_from_file_path = read_from_file_config["topology_file_path"];

            auto scale_free_network_config = config["scale_free_network"];
            enable_scale_free_network = scale_free_network_config["enable"];
            scale_free_network_gamma = scale_free_network_config["gamma"];
            scale_free_network_min_peer = scale_free_network_config["min_peer"];
            scale_free_network_buffer_to_peer_ratio = scale_free_network_config["buffer_to_peer_ratio"];
            scale_free_network_interval = scale_free_network_config["interval"];

            auto connection_pair_swap_config = config["connection_pair_swap"];
            enable_connection_pair_swap_config = connection_pair_swap_config["enable"];
            connection_pair_swap_percentage = connection_pair_swap_config["percentage"];
            connection_pair_swap_interval = connection_pair_swap_config["interval"];

        }

        LOG_IF(FATAL, enable_read_from_file) << "read from file is not implemented yet";

        this->enable = enable_read_from_file || enable_scale_free_network || enable_connection_pair_swap_config;
        if (static_cast<int>(enable_read_from_file) + static_cast<int>(enable_scale_free_network) + static_cast<int>(enable_connection_pair_swap_config) > 1)
        {
            LOG(FATAL) << "cannot enable multiple network_topology_manager services";
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path &output_path, std::map<std::string, node<model_datatype> *> &_node_container, std::vector<node<model_datatype> *> &_node_vector_container) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        this->set_node_container(_node_container, _node_vector_container);
        this->output_file_path = output_path / "network_topology_record.json";
        last_topology_update_tick = 0;
        output_file.open(output_file_path);
        output_file << "{\n";

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::end_of_tick) return {service_status::skipped, "not service_trigger_type::end_of_tick"};

        ////read from file
        if (enable_read_from_file)
        {
            LOG(FATAL) << "[network_topology_manager]: \"read_from_file\" not implement";
        }

        ////scale free network
        if (enable_scale_free_network)
        {
            if (tick >= last_topology_update_tick + scale_free_network_interval)
            {
                update_scale_free_topology(tick);
                last_topology_update_tick = tick;
            }
        }

        ////connection pair swap
        if (enable_connection_pair_swap_config)
        {
            if (tick >= last_topology_update_tick + connection_pair_swap_interval)
            {
                connection_pair_swap(tick);
                last_topology_update_tick = tick;
            }
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {
        output_file << "\n}\n";

        return {service_status::success, ""};
    }

private:
    void update_scale_free_topology(int tick)
    {
        int node_count = this->node_container->size();
        LOG_IF(FATAL, node_count < scale_free_network_min_peer + 1) << "not enough nodes to generate a scale free network";

        ////Scale-free network: https://en.wikipedia.org/wiki/Scale-free_network
        ////P(k) = A k ^ (-gamma)
        std::map<int, int> peer_count_per_node;
        std::map<int, double> weight_per_k;
        double total_weight = 0.0;
        for (int k = scale_free_network_min_peer; k < node_count - 1; ++k) // from peer=1 to peer=(node-1)
        {
            double current_weight = std::pow(k, -scale_free_network_gamma);
            weight_per_k[k] = current_weight;
            total_weight += current_weight;
        }
        for (auto& [k,weight]: weight_per_k)
        {
            weight /= total_weight;
        }

        std::vector<double> boundaries;
        boundaries.reserve(node_count);
        double accumulate_boundary = 0.0;
        for (auto &[k, weight]: weight_per_k)
        {
            boundaries.push_back(accumulate_boundary);
            accumulate_boundary += weight;
        }
        boundaries.push_back(accumulate_boundary);

        std::vector<std::tuple<int, int>> connections;
        int generate_degree_count = 1;
        while (generate_degree_count++)
        {
            static std::random_device rd;
            static std::mt19937 engine(rd());
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            while (true)
            {
                for (int node = 0; node < node_count; ++node)
                {
                    double random_number = distribution(engine);
                    int select_degree = 0;
                    for (int i = 0; i < boundaries.size() - 1; ++i)
                    {
                        if (boundaries[i] <= random_number && random_number < boundaries[i + 1])
                        {
                            select_degree = i + scale_free_network_min_peer;
                        }
                    }
                    peer_count_per_node[node] = select_degree;
                }

                ////check if it is possible
                size_t total_connection_terminals = 0;
                for (const auto &[node, peer_count]: peer_count_per_node)
                {
                    total_connection_terminals += peer_count;
                }
                if (total_connection_terminals % 2 == 0)
                    break;
            }

            ////begin generating the network topology
            int try_count = 0;
            bool whole_success = false;
            while (try_count < 10000)
            {
                try_count++;

                auto connection_optional = generate_network_topology(peer_count_per_node);
                if (connection_optional.has_value()) //success
                {
                    connections = *connection_optional;

                    ////check whether we get a network with islands
                    std::map<int, std::set<int>> peer_map;
                    for (auto &[node, peer]: connections)
                    {
                        peer_map[node].emplace(peer);
                        peer_map[peer].emplace(node);
                    }

                    ////check islands
                    std::set<int> mainland;
                    add_to_mainland(peer_map.begin()->first, peer_map, mainland);
                    if (mainland.size() == peer_map.size())
                    {
                        whole_success = true;
                        break;
                    }
                }
            }
            if (whole_success) break;
        }

        ////update model buffer size
        for (const auto &[node, peer_count]: peer_count_per_node)
        {
            auto node_iter = this->node_container->find(std::to_string(node));
            node_iter->second->buffer_size = static_cast<int>(peer_count * this->scale_free_network_buffer_to_peer_ratio);
        }

        std::map<int, std::set<int>> peer_per_node;
        for (auto& [lhs, rhs]: connections)
        {
            peer_per_node[lhs].emplace(rhs);
            peer_per_node[rhs].emplace(lhs);
        }

        ////update peers
        for (auto& [node, peers]: peer_per_node)
        {
            auto node_iter = this->node_container->find(std::to_string(node));
            node_iter->second->peers.clear();
            for (auto& single_peer: peers)
            {
                auto peer_name_str = std::to_string(single_peer);
                auto peer_iter = this->node_container->find(peer_name_str);
                node_iter->second->peers.emplace(peer_name_str, peer_iter->second);
            }
        }

        ////record the topology to json file
        nlohmann::json output = nlohmann::json::object();
        for (auto& [node, peers]: peer_per_node)
        {
            std::vector<std::string> peers_for_single_node;
            for (auto& single_peer: peers)
            {
                peers_for_single_node.push_back(std::to_string(single_peer));
            }
            nlohmann::json peers_for_single_node_json = peers_for_single_node;
            output[std::to_string(node)] = peers_for_single_node_json;
        }
        static bool first_json_object = true;
        if (first_json_object)
        {
            output_file << "\"tick-" << std::to_string(tick) << "\":" << output.dump();
            first_json_object = false;
        }
        else
        {
            output_file << ",\n" << "\"tick-" << std::to_string(tick) << "\":" << output.dump();
        }

    }

    void connection_pair_swap(int tick)
    {
        //get all connection pair
        std::vector<std::pair<std::string, std::string>> all_connections;
        {
            std::set<std::pair<std::string, std::string>> all_connections_set;
            for (const auto& [node_name, node] : *(this->node_container)) {
                for (const auto& [peer_name, peer_node] : node->peers) {
                    int node_name_int = std::stoi(node_name);
                    int peer_name_int = std::stoi(peer_name);
                    auto node_small = node_name_int < peer_name_int ? node_name : peer_name;
                    auto node_large = node_name_int > peer_name_int ? node_name : peer_name;
                    all_connections_set.insert(std::make_pair(node_small, node_large));
                }
            }
            for (const auto& target: all_connections_set) {
                all_connections.push_back(target);
            }
        }


        //find pairs to swap peers
        std::vector<std::pair<size_t, size_t>> swap_pairs;
        {
            size_t target_swap_count = all_connections.size() * this->connection_pair_swap_percentage / 2;
            std::set<size_t> available_connection_index;
            for (int i = 0; i < all_connections.size(); ++i) {
                available_connection_index.insert(i);
            }

            size_t find_suitable_swap_pair_count = 0;
            while (find_suitable_swap_pair_count < target_swap_count)
            {
                auto select_0 = util::select_randomly(available_connection_index.begin(), available_connection_index.end());
                auto select_1 = util::select_randomly(available_connection_index.begin(), available_connection_index.end());
                if (select_0 == select_1) continue; //same pick
                auto pair_0_lhs = all_connections[*select_0].first;
                auto pair_0_rhs = all_connections[*select_0].second;
                auto pair_1_lhs = all_connections[*select_1].first;
                auto pair_1_rhs = all_connections[*select_1].second;
                std::set<std::string> duplicate_check;
                duplicate_check.insert(pair_0_lhs);duplicate_check.insert(pair_0_rhs);
                duplicate_check.insert(pair_1_lhs);duplicate_check.insert(pair_1_rhs);
                if (duplicate_check.size() != 4) continue; //duplicate node
                
                auto connection_0_lhs_node = this->node_container->find(all_connections[*select_0].first)->second;
                auto connection_0_rhs_node = this->node_container->find(all_connections[*select_0].second)->second;
                auto connection_1_lhs_node = this->node_container->find(all_connections[*select_1].first)->second;
                auto connection_1_rhs_node = this->node_container->find(all_connections[*select_1].second)->second;
                
                if (connection_0_lhs_node->peers.find(pair_1_rhs) != connection_0_lhs_node->peers.end()) continue; //the target node is already connected.
                if (connection_1_lhs_node->peers.find(pair_0_lhs) != connection_0_lhs_node->peers.end()) continue; //the target node is already connected.

                find_suitable_swap_pair_count++;
                swap_pairs.emplace_back(*select_0, *select_1);

                available_connection_index.erase(select_0);
                available_connection_index.erase(select_1);
            }
        }

        //swap peers
        for (const auto& [connection_0, connection_1]: swap_pairs) {
            auto connection_0_lhs_node = this->node_container->find(all_connections[connection_0].first)->second;
            auto connection_0_rhs_node = this->node_container->find(all_connections[connection_0].second)->second;
            auto connection_1_lhs_node = this->node_container->find(all_connections[connection_1].first)->second;
            auto connection_1_rhs_node = this->node_container->find(all_connections[connection_1].second)->second;

            //remove connections
            auto erase_connections = [](node<model_datatype> * node, std::string erase_node_name){
                auto erase_iter = node->peers.find(erase_node_name);
                LOG_IF(FATAL, erase_iter == node->peers.end()) << "logic error";
                node->peers.erase(erase_iter);
            };
            {
                erase_connections(connection_0_lhs_node, connection_0_rhs_node->name);
                erase_connections(connection_0_rhs_node, connection_0_lhs_node->name);
                erase_connections(connection_1_lhs_node, connection_1_rhs_node->name);
                erase_connections(connection_1_rhs_node, connection_1_lhs_node->name);
            }

            //add connections
            auto add_connections = [this](node<model_datatype> * node, std::string add_node_name){
                auto add_iter = this->node_container->find(add_node_name);
                LOG_IF(FATAL, add_iter == this->node_container->end()) << "logic error";
                node->peers.emplace(add_iter->first, add_iter->second);
            };
            {
                add_connections(connection_0_lhs_node, connection_1_rhs_node->name);
                add_connections(connection_0_rhs_node, connection_1_lhs_node->name);
                add_connections(connection_1_lhs_node, connection_0_rhs_node->name);
                add_connections(connection_1_rhs_node, connection_0_lhs_node->name);
            }

            //record
            output_file << "\"tick-" << std::to_string(tick) << "\":" << "\"" <<
                    "{" << all_connections[connection_0].first << "--" << all_connections[connection_0].second << "}" <<
                    "{" << all_connections[connection_1].first << "--" << all_connections[connection_1].second << "}" <<
                    " ===>>> " <<
                    "{" << all_connections[connection_0].first << "--" << all_connections[connection_1].second << "}" <<
                    "{" << all_connections[connection_1].first << "--" << all_connections[connection_0].second << "}" <<
                    "\"," << std::endl;
        }
    }
};

template <typename model_datatype>
class delta_weight_after_training_averaging_record : public service<model_datatype>
{
public:
    //set these variables before init
    std::filesystem::path output_records_path;
    std::string path;
    std::vector<std::string> nodes_to_record;
    std::set<std::string> layers_to_record;

    delta_weight_after_training_averaging_record()
    {

    }

    std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
    {
        this->enable = config["enable"];
        this->path = config["path"];
        std::string nodes_to_record_str = config["nodes_to_record"];
        this->nodes_to_record = util::split(nodes_to_record_str, ',');
        std::string layers_to_record_str = config["layers_to_record"];
        const std::vector<std::string> layers = util::split(layers_to_record_str, ',');
        for (const auto& layer : layers) {
            this->layers_to_record.emplace(layer);
        }
        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
    {
        this->set_node_container(_node_container, _node_vector_container);

        if (this->enable == false) return {service_status::skipped, "not enabled"};

        //create the output records path or not?
        this->output_records_path = output_path / this->path;
        if (not std::filesystem::exists(this->output_records_path))
            std::filesystem::create_directories(this->output_records_path);

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::start_of_tick && tick == 0) {
            //add nodes
            for (int i = 0; i < this->node_vector_container->size(); ++i) {
                const auto current_node = (*this->node_vector_container)[i];
                const auto& current_node_name = current_node->name;
                if (std::find(this->nodes_to_record.begin(), this->nodes_to_record.end(), current_node_name) == this->nodes_to_record.end()) continue; //skip if this node is not required to record
                const Ml::caffe_parameter_net<model_datatype>& current_parameter = current_node->solver->get_parameter();
                this->current_parameters[current_node->name] = current_parameter;

                //creat output file
                std::shared_ptr<std::ofstream> temp_file = std::make_shared<std::ofstream>();
                temp_file->open(this->output_records_path / (current_node_name + ".csv"), std::ios::binary);
                //create the header
                const auto& current_model = current_node->solver->get_parameter();
                create_csv_header(temp_file, current_model);
                //store the init state
                store_weight_to_csv_row(temp_file, "init", tick, current_model);
                this->output_delta_weight_files[current_node->name] = temp_file;
            }
        }

        if (trigger == service_trigger_type::end_of_training) {
            tmt::ParallelExecution([&tick, this](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node) {
                if (!single_node->model_trained) return;    //return if the node is not trained for this tick
                if (std::find(this->nodes_to_record.begin(), this->nodes_to_record.end(), single_node->name) == this->nodes_to_record.end()) return; //skip if this node is not required to record

                const auto current_parameter_iter = this->current_parameters.find(single_node->name);
                if (current_parameter_iter == this->current_parameters.end()) LOG(FATAL) << "bug in delta_weight_after_training_record: " + single_node->name + " not in the this->current_parameters";

                //calculate delta weight
                const auto& old_model = current_parameter_iter->second;
                const auto& current_model = single_node->solver->get_parameter();
                const auto delta = current_model - old_model;
                current_parameter_iter->second = current_model;

                //store delta
                std::shared_ptr<std::ofstream> file_ptr = this->output_delta_weight_files[single_node->name];
                store_weight_to_csv_row(file_ptr, "train", tick, delta);
            }, this->node_vector_container->size(), this->node_vector_container->data());
        }

        if (trigger == service_trigger_type::end_of_averaging) {
            tmt::ParallelExecution([&tick, this](uint32_t index, uint32_t thread_index, node<model_datatype> *single_node) {
                if (!single_node->model_averaged) return;    //return if the node is not averaged for this tick
                if (std::find(this->nodes_to_record.begin(), this->nodes_to_record.end(), single_node->name) == this->nodes_to_record.end()) return; //skip if this node is not required to record

                const auto current_parameter_iter = this->current_parameters.find(single_node->name);
                if (current_parameter_iter == this->current_parameters.end()) LOG(FATAL) << "bug in delta_weight_after_training_averaging_record: " + single_node->name + " not in the this->current_parameters";

                //calculate delta weight
                const auto& old_model = current_parameter_iter->second;
                const auto& current_model = single_node->solver->get_parameter();
                const auto delta = current_model - old_model;
                current_parameter_iter->second = current_model;

                //store delta
                std::shared_ptr<std::ofstream> file_ptr = this->output_delta_weight_files[single_node->name];
                store_weight_to_csv_row(file_ptr, "average", tick, delta);
            }, this->node_vector_container->size(), this->node_vector_container->data());
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {
        for (auto& [key, file]: output_delta_weight_files) {
            file->flush();
            file->close();
        }
        return {service_status::success, ""};
    }

private:
    std::map<std::string, std::shared_ptr<std::ofstream>> output_delta_weight_files;
    std::map<std::string, Ml::caffe_parameter_net<model_datatype>> current_parameters;

    void create_csv_header(std::shared_ptr<std::ofstream> file, const Ml::caffe_parameter_net<model_datatype>& model) {
        *file << "tick" << "," << "type";
        for (const Ml::caffe_parameter_layer<model_datatype>& layer : model.getLayers()) {
            const auto& layer_name = layer.getName();
            if ( !(layers_to_record.contains(layer_name) || layers_to_record.contains("all")) ) {
                continue;
            }

            const size_t layer_size = layer.size();
            if (layer_size == 0) continue;
            for (size_t j = 0; j < layer_size; ++j) {
                *file << "," << layer_name + "-" + std::to_string(j);
            }
            *file << "," << "distance" << "+" << layer_name;
            for (size_t j = 0; j < layer_size; ++j) {
                *file << "," << layer_name + "-" + std::to_string(j) + "-" + "angle";
            }
        }
        *file << std::endl;
    }

    void store_weight_to_csv_row(std::shared_ptr<std::ofstream> file, std::string type, int tick, const Ml::caffe_parameter_net<model_datatype>& model) {
        *file << tick << "," << type;
        for (const Ml::caffe_parameter_layer<model_datatype>& layer : model.getLayers()) {
            const auto& layer_name = layer.getName();
            if ( !(layers_to_record.contains(layer_name) || layers_to_record.contains("all")) ) {
                continue;
            }

            const auto layer_size = layer.size();
            if (layer_size == 0) continue;
            const auto& blobs = layer.getBlob_p();
            model_datatype distance_sum = 0;
            for (int i = 0; i < blobs.size(); ++i) {
                const auto& data = blobs[i]->getData();
                for (const auto& v : data) {
                    *file << "," << v;
                    distance_sum += v*v;
                }
            }
            //store distance
            model_datatype distance = std::sqrt(distance_sum);
            *file << "," << distance;
            //calculate angle
            for (int i = 0; i < blobs.size(); ++i) {
                const auto& data = blobs[i]->getData();
                for (const auto& v : data) {
                    *file << "," << v / distance;
                }
            }
        }
        *file << std::endl;
    }

};

template <typename model_datatype>
class delta_weight_item
{
public:
    delta_weight_item()
    {
        apply_tick = 0;
        apply_type = "";
    }

    int apply_tick;
    std::string apply_type;
    Ml::caffe_parameter_net<model_datatype> delta_weight;
};

template <typename model_datatype>
class apply_delta_weight : public service<model_datatype>
{
private:
    std::map<std::string, std::shared_ptr<std::ifstream>> delta_weight_file;
    std::map<std::string, delta_weight_item<model_datatype>> delta_weight_record;
    std::map<std::string, std::set<std::string>> node_enabled_item;
    std::map<std::string, std::vector<std::tuple<std::string, size_t>>> map_to_weight_pos;
public:
    const std::string SKIP_COLUMN_CONTENT = "!SKIP";
    const std::set<std::string> UPDATE_TYPE_OVERRIDE = {"init"};
    const std::set<std::string> UPDATE_TYPE_DELTA = {"average", "train"};

public:
    apply_delta_weight()
    {
        this->node_vector_container = nullptr;
    }

    std::tuple<service_status, std::string> apply_config(const configuration_file::json &config) override
    {
        this->enable = config["enable"];
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        //load config
        const std::string config_str = config["config"];
        auto node_apply_type_pairs = extract_node_and_apply_type(config_str);
        for (const auto& [node, apply_type] : node_apply_type_pairs) {
            const auto trim_node = trim(node);
            const auto trim_apply_type = trim(apply_type);
            node_enabled_item[trim_node].insert(trim_apply_type);
        }

        //load first line of csv
        for (const auto& [node, enabled_item] : node_enabled_item)
        {
            const std::filesystem::path csv_path = std::filesystem::current_path() / (node + ".csv");
            if (!std::filesystem::exists(csv_path))
                LOG(FATAL) << csv_path.string() << " does not exist";

            auto file = std::make_shared<std::ifstream>(csv_path, std::ios::in | std::ios::binary);
            LOG_IF(FATAL, !file->is_open()) << "cannot open " << csv_path.string();
            delta_weight_file.emplace(node, file);

            //read the first line
            std::string header_line;
            std::getline(*file, header_line);

            //process the csv file
            size_t current_pos = -1;
            current_pos = header_line.find_first_of(',', current_pos+1); //skip the first ','
            current_pos = header_line.find_first_of(',', current_pos+1);  //skip the "type" column
            //record the column now
            std::vector<std::tuple<std::string, size_t>> column_record;
            while (true)
            {
                size_t next_pos = header_line.find_first_of(',', current_pos+1);
                if (next_pos == std::string::npos) break;
                const std::string current_str = header_line.substr(current_pos+1, next_pos-current_pos-1);
                current_pos = next_pos;
                size_t dash_pos = current_str.find_first_of('-');
                if (dash_pos == std::string::npos)
                {
                    column_record.emplace_back(SKIP_COLUMN_CONTENT, 0);
                    continue; //this is not a delta weight, maybe something else
                }
                const std::string layer_name = current_str.substr(0, dash_pos);
                const size_t weight_pos = std::stoull(current_str.substr(dash_pos+1));
                column_record.emplace_back(layer_name, weight_pos);
            }
            map_to_weight_pos[node] = column_record;
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path &output_path, std::map<std::string, node<model_datatype> *> &_node_container, std::vector<node<model_datatype> *> &_node_vector_container) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        this->set_node_container(_node_container, _node_vector_container);

        //read the next line of csv
        for (const auto& [node, enabled_item] : node_enabled_item)
        {
            std::shared_ptr<std::ifstream> file = this->delta_weight_file[node];

            auto result = find_next_line(file, enabled_item);
            if (!result) continue;  //the next line is not found

            const auto [apply_tick, apply_type, current_line] = *result;
            auto delta_model = convert_line_to_delta_weight(current_line, node);

            delta_weight_item<model_datatype> apply_item;
            apply_item.delta_weight = delta_model;
            apply_item.apply_type = apply_type;
            apply_item.apply_tick = apply_tick;

            delta_weight_record[node] = apply_item;
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (this->enable == false) return {service_status::skipped, "not enabled"};

        if (trigger != service_trigger_type::end_of_tick) return {service_status::skipped, "not service_trigger_type::end_of_tick"};

        for (auto& [node_name, delta_weight] : this->delta_weight_record) {
            const auto node_iter = this->node_container->find(node_name);
            if (node_iter == this->node_container->end()) LOG(FATAL) << node_name << " specified in apply_delta_weight does not exist";
            const auto& old_model = node_iter->second->solver->get_parameter();
            int apply_tick = delta_weight.apply_tick;
            std::string apply_type = delta_weight.apply_type;
            const Ml::caffe_parameter_net<model_datatype>& delta = delta_weight.delta_weight;

            //apply delta weight
            if (tick == apply_tick) {
                LOG(INFO) << "tick:" << tick << ", apply delta weight, node:" << node_name << ", type:" << apply_type;
                if (this->UPDATE_TYPE_DELTA.contains(apply_type))
                {
                    const auto& new_model = old_model + delta;
                    node_iter->second->solver->set_parameter(new_model);
                }
                else if (this->UPDATE_TYPE_OVERRIDE.contains(apply_type))
                {
                    const auto& new_model = delta;
                    node_iter->second->solver->set_parameter(new_model);
                }
                else
                {
                    LOG(FATAL) << "unknown apply type in apply delta weight service: " << apply_type;
                }

                std::shared_ptr<std::ifstream> file = this->delta_weight_file[node_name];
                auto enabled_item = this->node_enabled_item[node_name];
                auto result = find_next_line(file, enabled_item);
                if (!result) continue;  //the next line is not found

                const auto [next_apply_tick, next_apply_type, current_line] = *result;
                auto next_delta_model = convert_line_to_delta_weight(current_line, node_name);

                delta_weight_item<model_datatype> apply_item;
                apply_item.delta_weight = next_delta_model;
                apply_item.apply_type = next_apply_type;
                apply_item.apply_tick = next_apply_tick;
                delta_weight = apply_item;
            }
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {
        return {service_status::success, ""};
    }

private:
    std::vector<std::pair<std::string, std::string>> extract_node_and_apply_type(const std::string& s) {
        std::vector<std::pair<std::string, std::string>> result;
        std::istringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            std::istringstream itemStream(item);
            std::string numberStr, str;
            if (std::getline(itemStream, numberStr, ':') && std::getline(itemStream, str)) {
                result.emplace_back(numberStr, str);
            }
        }
        return result;
    }

    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(' ');
        if (std::string::npos == first) {
            return str;
        }
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }

    std::tuple<size_t, std::string> find_next_csv_item(const std::string& target_string, char target, size_t pos) {
        size_t next_pos = target_string.find_first_of(',', pos);
        const std::string output_str = target_string.substr(pos, next_pos - pos);
        return std::make_tuple(next_pos, output_str);
    }

    std::optional<std::tuple<size_t, std::string, std::string>> find_next_line(std::shared_ptr<std::ifstream> file, const std::set<std::string>& enabled_types) {
        while (true) {
            std::string current_line;
            std::getline(*file, current_line);

            if (file->eof()) return std::nullopt;
            LOG_IF(FATAL, file->bad()) << "error when reading delta weight file";

            //read the first two column
            size_t current_pos = 0, next_pos = 0;
            next_pos = current_line.find_first_of(',', current_pos); //find the tick
            std::string tick_str = current_line.substr(current_pos, next_pos-current_pos);
            current_pos = next_pos + 1;

            next_pos = current_line.find_first_of(',', current_pos);  //find the type
            std::string type_str = current_line.substr(current_pos, next_pos-current_pos);
            current_pos = next_pos + 1;

            if (enabled_types.contains(type_str))
                return {{std::stoull(tick_str), type_str, current_line.substr(current_pos)}};
        }
    }

    Ml::caffe_parameter_net<model_datatype> convert_line_to_delta_weight(const std::string& line, const std::string& node) {
        auto node_iter = this->node_container->find(node);
        auto ml_model = node_iter->second->solver->get_parameter();
        ml_model.set_all(0);
        size_t current_pos = 0;
        size_t column_index = 0;
        while (true) {
            const auto [next_pos, str_item] = find_next_csv_item(line, ',', current_pos);
            if (next_pos == std::string::npos) break;
            current_pos = next_pos + 1;

            //find the location
            const auto& map_to_weight = this->map_to_weight_pos[node];
            const auto [layer_name, weight_index] = map_to_weight[column_index];
            column_index++;
            if (layer_name==SKIP_COLUMN_CONTENT) continue;

            LOG_IF(FATAL, node_iter == this->node_container->end()) << node << " specified in apply delta weight service does not exist";
            auto& layers = ml_model.getLayers();
            std::vector<boost::shared_ptr<Ml::tensor_blob_like<model_datatype>>> blob;
            for (auto& layer : layers) {
                if (layer.getName() == layer_name)
                {
                    blob = layer.getBlob_p();
                    break;
                }
            }
            LOG_IF(FATAL, blob.empty()) << layer_name << " not found in the ML layer";
            //calculate actual weight position
            size_t remain_weight_index = weight_index, blob_index=0;
            while (true) {
                if (remain_weight_index < blob[blob_index]->size()) {
                    break;
                }
                else {
                    remain_weight_index -= blob[blob_index]->size();
                    blob_index++;
                }
            }
            if constexpr (std::is_same_v<model_datatype, float>) {
                float v = std::stof(str_item);
                blob[blob_index]->getData()[remain_weight_index] = v;
            }
            if constexpr (std::is_same_v<model_datatype, double>) {
                double v = std::stod(str_item);
                blob[blob_index]->getData()[remain_weight_index] = v;
            }
        }

        return ml_model;
    }

};


template <typename model_datatype>
class stage_manager_service : public service<model_datatype>
{
public:
    //set these variables before init

private:
    std::filesystem::path stage_script;
    std::map<int, std::vector<std::map<std::string, configuration_file::json>>> all_scripts;
    std::map<std::string, bool> enabled_status;

public:
    stage_manager_service()
    {

    }

    std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
    {
        this->enable = config["enable"];
        this->stage_script = std::filesystem::path(config["script_path"]);

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
    {
        this->set_node_container(_node_container, _node_vector_container);

        if (!this->enable) return {service_status::skipped, "not enabled"};

        LOG_IF(FATAL, !std::filesystem::exists(stage_script)) << "stage script file path " << stage_script.string() << " does not exist.";
        std::ifstream script_file(stage_script, std::ios::binary);
        LOG_ASSERT(script_file.is_open());
        std::string content;
        content.assign((std::istreambuf_iterator<char>(script_file)),(std::istreambuf_iterator<char>()));
        script_file.close();

        //backup stage script file
        {
            std::ofstream backup_script_file(output_path / stage_script.filename(), std::ios::binary);
            LOG_ASSERT(backup_script_file.is_open());
            backup_script_file << content;
            backup_script_file.close();
        }

        configuration_file::json script_file_json = configuration_file::json::parse(content);
        LOG_ASSERT(script_file_json.is_array());
        for (const auto& single_script : script_file_json) {
            int tick = single_script["tick"].get<int>();
            std::map<std::string, configuration_file::json> scripts_at_this_tick;
            for (const auto& [name, actions] : single_script["script"].items()) {
                scripts_at_this_tick[name] = actions;
            }
            all_scripts[tick].push_back(scripts_at_this_tick);
        }
        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        if (!this->enable) return {service_status::skipped, "not enabled"};

        if (trigger == service_trigger_type::start_of_tick) {
            const auto tick_find_iter = all_scripts.find(tick);
            if (tick_find_iter == all_scripts.end()) {
                return {service_status::skipped, "not time yet"};
            }
            const auto& node_and_script_vec = tick_find_iter->second;
            for (const auto& node_and_script : node_and_script_vec) {
                for (const auto& [node_name, script_item_json] : node_and_script) {
                    if (node_name == "remove_edge") {   //special operations
                        for (const auto& edge : script_item_json) {
                            const std::string edge_str = edge;
                            {
                                const size_t index = edge_str.find("--");
                                if (index != std::string::npos) {
                                    std::string node_0_name = edge_str.substr(0, index);
                                    std::string node_1_name = edge_str.substr(index+2);
                                    remove_peer_of_node(node_0_name, node_1_name);
                                    remove_peer_of_node(node_1_name, node_0_name);
                                }
                            }
                            {
                                const size_t index = edge_str.find("->");
                                if (index != std::string::npos) {
                                    std::string node_0_name = edge_str.substr(0, index);
                                    std::string node_1_name = edge_str.substr(index+2);
                                    remove_peer_of_node(node_0_name, node_1_name);
                                }
                            }
                        }
                    }
                    else if (node_name == "add_edge") { //special operations
                        for (const auto& edge : script_item_json) {
                            const std::string edge_str = edge;
                            {
                                const size_t index = edge_str.find("--");
                                if (index != std::string::npos) {
                                    std::string node_0_name = edge_str.substr(0, index);
                                    std::string node_1_name = edge_str.substr(index+2);
                                    add_peer_of_node(node_0_name, node_1_name);
                                    add_peer_of_node(node_1_name, node_0_name);
                                }
                            }
                            {
                                const size_t index = edge_str.find("->");
                                if (index != std::string::npos) {
                                    std::string node_0_name = edge_str.substr(0, index);
                                    std::string node_1_name = edge_str.substr(index+2);
                                    add_peer_of_node(node_0_name, node_1_name);
                                }
                            }
                        }
                    }
                    else if (node_name == "only_enable") { //special operations
                        //disable all nodes at beginning
                        enabled_status.clear();
                        for (auto& [name, node] : *(this->node_container)) {
                            enabled_status[name] = node->enable;
                            node->enable = false;
                        }
                        //enable selected nodes
                        for (const auto& enabled_node_name : script_item_json) {
                            const std::string enabled_node_name_str = enabled_node_name;
                            auto node_iter = this->node_container->find(enabled_node_name_str);
                            LOG_ASSERT(node_iter != this->node_container->end()) << "info: " << enabled_node_name_str << " does not exist";
                            auto& target_node = node_iter->second;
                            target_node->enable = true;
                        }
                    }
                    else {
                        //simple node names
                        auto node_iter = this->node_container->find(node_name);
                        LOG_ASSERT(node_iter != this->node_container->end()) << "info: " << node_name << " does not exist";
                        auto& target_node = node_iter->second;
                        apply_config_to_node(script_item_json, target_node, this->node_container, this->node_vector_container);
                    }
                }
            }
        }
        else if (trigger == service_trigger_type::end_of_tick) {
            const auto tick_find_iter = all_scripts.find(tick);
            if (tick_find_iter == all_scripts.end()) {
                return {service_status::skipped, "not time yet"};
            }
            const auto& node_and_script_vec = tick_find_iter->second;
            for (const auto& node_and_script : node_and_script_vec) {
                for (const auto& [node_name, script_item_json] : node_and_script) {
                    if (node_name == "only_enable") { //special operations
                        //recover all status
                        for (auto& [name, node] : *(this->node_container)) {
                            node->enable = enabled_status[name];
                        }
                    }
                    else {
                        //other stuffs that would not work at end_of_tick
                    }
                }
            }
        }
        else {
            return {service_status::skipped, "not start_of_tick or end_of_tick"};
        }



        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {

        return {service_status::success, ""};
    }

private:
    void apply_config_to_node(const configuration_file::json& script, node<model_datatype>* target_node, std::map<std::string, node<model_datatype>*>* node_container, std::vector<node<model_datatype>*>* node_vector_container) {
        if (script.contains("enable")) {
            target_node->enable = script["enable"].get<bool>();
        }
        if (script.contains("buffer_size")) {
            target_node->buffer_size = script["buffer_size"].get<int>();
        }
        if (script.contains("training_interval_tick")) {
            target_node->training_interval_tick = script["training_interval_tick"].get<std::vector<int>>();
        }
        if (script.contains("override_model_from")) {
            const std::string src_node_name = script["override_model_from"].get<std::string>();
            auto iter = node_container->find(src_node_name);
            LOG_ASSERT(iter != node_container->end());
            const auto& src_model = iter->second->solver->get_parameter();
            target_node->solver->set_parameter(src_model);
        }
        if (script.contains("node_type")) {
            const std::string new_target_node_type = script["node_type"].get<std::string>();
            const std::string target_node_name = target_node->name;
            auto iter = node_container->find(target_node_name);
            LOG_ASSERT(iter != node_container->end());

            //find the node in the registered node map
            node<model_datatype> *temp_node = nullptr;
            {
                auto result = node<model_datatype>::get_node_by_type(new_target_node_type);
                if (result == nullptr) {
                    LOG(FATAL) << "unknown node type:" << new_target_node_type;
                } else {
                    temp_node = result->new_node(iter->second->name, iter->second->buffer_size, iter->second->node_type_arg);
                }
            }
            //move properties of old node to new node
            iter->second->copy_properties_to_new_node(temp_node);

            //update node_vector_container
            for (int i = 0; i < this->node_vector_container->size(); ++i) {
                auto& node_i = this->node_vector_container->at(i);
                if (node_i->name == temp_node->name) {
                    node_i = temp_node;
                }
            }

            //update node_container
            delete iter->second;
            iter->second = temp_node;

            //update peers and planned_peers container
            for (auto node_container_iter = node_container->begin(); node_container_iter != node_container->end(); ++node_container_iter) {
                if (node_container_iter->second->peers.contains(temp_node->name)) {
                    node_container_iter->second->peers[temp_node->name] = temp_node;
                }
                if (node_container_iter->second->planned_peers.contains(temp_node->name)) {
                    node_container_iter->second->planned_peers[temp_node->name] = temp_node;
                }
            }
        }
        if (script.contains("set_weights")) {
            const model_datatype target_weight_value = script["set_weights"].get<model_datatype>();
            const std::string target_node_name = target_node->name;
            auto iter = node_container->find(target_node_name);
            LOG_ASSERT(iter != node_container->end());
            auto model = iter->second->solver->get_parameter();
            model.set_all(target_weight_value);
            iter->second->solver->set_parameter(model);
        }
        if (script.contains("scale_weights")) {
            const model_datatype scale_value = script["scale_weights"].get<model_datatype>();
            const std::string target_node_name = target_node->name;
            auto iter = node_container->find(target_node_name);
            LOG_ASSERT(iter != node_container->end());
            auto model = iter->second->solver->get_parameter();
            model = model * scale_value;
            iter->second->solver->set_parameter(model);
        }
    }

    bool remove_peer_of_node(const std::string& node_name, const std::string& peer_name) {
        auto node_iter = this->node_container->find(node_name);
        const auto peer_iter = node_iter->second->peers.find(peer_name);
        if (peer_iter == node_iter->second->peers.end()) {
            return false;
        }
        node_iter->second->peers.erase(peer_iter);
        return true;
    }

    bool add_peer_of_node(const std::string& node_name, const std::string& peer_name) {
        auto node_iter = this->node_container->find(node_name);
        const auto peer_iter = node_iter->second->peers.find(peer_name);
        if (peer_iter != node_iter->second->peers.end()) {
            return false;
        }
        const auto peer_inst_iter = this->node_container->find(peer_name);
        node_iter->second->peers.emplace(peer_name, peer_inst_iter->second);
        return true;
    }
};

template <typename model_datatype>
class compiled_services : public service<model_datatype>{
public:
    static constexpr bool ENABLE_SET_NODE_0_WEIGHT = false;
    static constexpr bool ENABLE_TRAIN_0_4_FIRST_10000_TICK_5_9_LATER_10000_TICK = false;

    bool enable_model_randomness;
    float init_model_randomness;


    std::tuple<service_status, std::string> apply_config(const configuration_file::json& config) override
    {
        enable_model_randomness = config["enable_model_randomness"];
        init_model_randomness = config["init_model_randomness"];

        return {service_status::success, ""};   //compiled services cannot have config
    }

    std::tuple<service_status, std::string> init_service(const std::filesystem::path& output_path, std::map<std::string, node<model_datatype> *>& _node_container, std::vector<node<model_datatype>*>& _node_vector_container) override
    {
        this->set_node_container(_node_container, _node_vector_container);

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_per_tick(int tick, service_trigger_type trigger) override
    {
        // set all weights of node 0 to a value at tick 0
        if (ENABLE_SET_NODE_0_WEIGHT && tick == 0 && trigger == service_trigger_type::start_of_tick) {
            LOG(WARNING) << "ENABLE_SET_NODE_0_WEIGHT is true, the model weight of node 0 will be set to specific values";

            const auto node_0_iter = this->node_container->find("0");
            LOG_IF(FATAL, node_0_iter == this->node_container->end()) << "node 0 does not exist";
            Ml::caffe_parameter_net<model_datatype> model = node_0_iter->second->solver.get()->get_parameter();
            std::vector<Ml::caffe_parameter_layer<model_datatype>>& layers = model.getLayers();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-0.001,0.001);
            for (auto& layer : layers) {
                for (boost::shared_ptr<Ml::tensor_blob_like<model_datatype>> blob : layer.getBlob_p()) {
                    auto& tensor_data = blob->getData();
                    for (auto& data : tensor_data) {
                        data = dis(gen);
                    }
                }
            }
            node_0_iter->second->solver.get()->set_parameter(model);
        }

        //override the initial models with a specific model + randomness
        if (enable_model_randomness && tick == 0 && trigger == service_trigger_type::start_of_tick) {
            LOG(WARNING) << "enable_model_randomness is true, setting the model to a mixture of node 0 and randomness model";

            const auto node_0_iter = this->node_container->find("0");
            LOG_IF(FATAL, node_0_iter == this->node_container->end()) << "node 0 does not exist";
            Ml::caffe_parameter_net<model_datatype> model_0 = node_0_iter->second->solver.get()->get_parameter();
            for (auto& node: *(this->node_container))
            {
                auto model = model_0 * (1-init_model_randomness) + node.second->solver->get_parameter() * init_model_randomness;
                node.second->solver->set_parameter(model);
            }
        }

        //ENABLE_TRAIN_0_4_FIRST_10000_TICK_5_9_LATER_10000_TICK
        if(ENABLE_TRAIN_0_4_FIRST_10000_TICK_5_9_LATER_10000_TICK) {
            if (tick == 0 && trigger == service_trigger_type::start_of_tick) {
                LOG(WARNING) << "ENABLE_TRAIN_0_4_FIRST_10000_TICK_5_9_LATER_10000_TICK is true, setting all nodes to normal_label_0_4";
                for (auto& node: *(this->node_container)) {
                    node.second->type = node_type::normal_label_0_4;
                }
            }
            if (tick == 10000 && trigger == service_trigger_type::start_of_tick) {
                LOG(WARNING) << "ENABLE_TRAIN_0_4_FIRST_10000_TICK_5_9_LATER_10000_TICK is true, setting all nodes to normal";
                for (auto& node: *(this->node_container)) {
                    node.second->type = node_type::normal;
                }
            }
        }

        return {service_status::success, ""};
    }

    std::tuple<service_status, std::string> process_on_event(int tick, service_trigger_type trigger, std::string triggered_node_name) override
    {
        LOG(FATAL) << "not implemented";
        return {service_status::fail_not_specified_reason, "not implemented"};
    }

    std::tuple<service_status, std::string> destruction_service() override
    {
        return {service_status::success, ""};
    }

private:
    void trigger_model_weight_override() {

    }
};
