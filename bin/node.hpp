//
// Created by jxzhang on 09-08-21. Malicious types extended by tyd on 20-09-21
//
#pragma once

#ifndef DFL_NODE_HPP
#define DFL_NODE_HPP

#include <optional>
#include <random>
#include <set>
#include <vector>
#include <unordered_map>
#include <ml_layer.hpp>

enum class dataset_mode_type
{
	unknown,
	default_dataset,
	iid_dataset,
	non_iid_dataset,
	
};

namespace std
{
    inline std::ostream& operator << (std::ostream& os, const dataset_mode_type& x)
    {
        switch (x)
        {
            case dataset_mode_type::unknown:
                os << "unknown";
                break;
            case dataset_mode_type::default_dataset:
                os << "default_dataset";
                break;
            case dataset_mode_type::iid_dataset:
                os << "iid_dataset";
                break;
            case dataset_mode_type::non_iid_dataset:
                os << "non_iid_dataset";
                break;
        }
        return os;
    }
    
    inline std::string to_string(const dataset_mode_type& x)
    {
        std::ostringstream ss;
        ss << x;
        return ss.str();
    }
}

enum node_type
{
	unknown_node_type = 0,
	normal,
	observer,       //node that only performs averaging, will not send models
    no_training,    //nodes will not perform training, but will send models
    pontificator,   //nodes will always propagate its initial model
	malicious_model_poisoning_random_model,             //always send random models
	malicious_model_poisoning_random_model_by_turn,     //send random models by turns(one good model + one malicious model)
	malicious_model_poisoning_random_model_biased_0_1,  //subtract 0-0.1 from the model weights
	malicious_duplication_attack,                       //duplicate other's model and send it to others
	malicious_data_poisoning_shuffle_label,             //shuffle the label of training dataset and send it as training dataset to nodes
	malicious_data_poisoning_shuffle_label_biased_1,    //add 1 to the label of training dataset and send it as training dataset to nodes
	malicious_data_poisoning_random_data,               //use uniformly random training dataset to train the model

    normal_label_0_4,   //normal nodes but only receive training dataset whose labels are from 0 to 4
    normal_label_5_9,   //normal nodes but only receive training dataset whose labels are from 5 to 9

    federated_learning_server,  //server do not training, only perform averaging and send averaged model to other nodes
    
    normal_reduced_sending, //normal node but only send models to others per x rounds. x can be specified by args.
    
	node_type_last_index
};

template<typename model_datatype>
class simulation_service_data_type
{
public:
    Ml::caffe_parameter_net<model_datatype>* just_received_model_ptr;
    std::string just_received_model_source_node_name;
    std::mutex just_received_model_ptr_lock;

    Ml::caffe_parameter_net<model_datatype>* average_output_ptr;

    Ml::caffe_parameter_net<model_datatype>* model_before_training_ptr;
    Ml::caffe_parameter_net<model_datatype>* model_after_training_ptr;
};

template<typename model_datatype>
class node
{
public:
	node(std::string _name, size_t buf_size, std::optional<std::string> arg = {}) : name(std::move(_name)), next_train_tick(0), buffer_size(buf_size), planned_buffer_size(buf_size), dataset_mode(dataset_mode_type::unknown), model_generation_type(Ml::model_compress_type::unknown), filter_limit(0.0f), last_measured_accuracy(0.0f), last_measured_tick(0), type(node_type::unknown_node_type), node_type_arg(arg)
	{
		solver.reset(new Ml::MlCaffeModel<model_datatype, caffe::SGDSolver>());
        enable = true;
        enable_averaging = true;
	}
	
	virtual ~node() = default;
	
	std::string name;
	dataset_mode_type dataset_mode;
	node_type type;
    std::optional<std::string> node_type_arg;
	//std::unordered_map<int, std::tuple<Ml::caffe_parameter_net<model_datatype>, float>> nets_record; //for delayed accuracy testing
	std::unordered_map<int, model_datatype> nets_accuracy_only_record; //for non-delayed accuracy testing
	int next_train_tick;
	std::unordered_map<int, std::tuple<float, float>> special_non_iid_distribution; //label:{min,max}
	std::vector<int> training_interval_tick;
	
	size_t buffer_size;
	size_t planned_buffer_size;
    bool enable;
    bool enable_averaging;
	
	std::vector<std::tuple<std::string, Ml::model_compress_type, Ml::caffe_parameter_net<model_datatype>>> parameter_buffer;
	std::mutex parameter_buffer_lock;
	std::shared_ptr<Ml::MlCaffeModel<model_datatype, caffe::SGDSolver>> solver;
	std::unordered_map<std::string, double> reputation_map;
	Ml::model_compress_type model_generation_type;
	float filter_limit;
	
	std::unordered_map<std::string, node *> peers;
	std::unordered_map<std::string, node *> planned_peers;
	
	float last_measured_accuracy;
	int last_measured_tick;
    bool model_trained;
    bool model_averaged;

    //This variable is only for simulation service purpose
    simulation_service_data_type<model_datatype> simulation_service_data;

    void copy_properties_to_new_node(node* target) {
//        target->name = name;//set by constructor
        target->dataset_mode = dataset_mode;
//        target->type = type;//set by constructor
        target->nets_accuracy_only_record = nets_accuracy_only_record;
        target->next_train_tick = next_train_tick;
        target->special_non_iid_distribution = special_non_iid_distribution;
        target->training_interval_tick = training_interval_tick;

//        target->buffer_size = buffer_size;//set by constructor
        target->planned_buffer_size = planned_buffer_size;
        target->enable = enable && target->enable;  //if enable is false, then we set output enable to false
        target->enable_averaging = enable_averaging && target->enable_averaging; //if enable_averaging is false, then we set output enable_averaging to false

        {
            std::lock_guard guard_src(parameter_buffer_lock);
            std::lock_guard guard_dst(target->parameter_buffer_lock);
            target->parameter_buffer = parameter_buffer;
            //target->parameter_buffer_lock = parameter_buffer_lock;//cannot copy mutex
        }

        target->solver = solver;
        target->reputation_map = reputation_map;
        target->model_generation_type = model_generation_type;
        target->filter_limit = filter_limit;

        target->peers = peers;
        target->planned_peers = planned_peers;

        target->name = name;

        target->last_measured_accuracy = last_measured_accuracy;
        target->last_measured_tick = last_measured_tick;
        target->model_trained = model_trained;
        target->model_averaged = model_averaged;
    }

	virtual void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) = 0;
    
    virtual float evaluate_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label)
    {
        float accuracy = this->solver->evaluation(data, label);
        return accuracy;
    }
	
	virtual std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() = 0;
	
	virtual node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) = 0;
	
	static node<model_datatype> *get_node_by_type(const std::string type)
	{
		auto iter = RegisteredNodeType.find(type);
		if (iter == RegisteredNodeType.end())
		{
			return nullptr;
		}
		else
		{
			return iter->second;
		}
	}
	
	static std::optional<node_type> get_node_type_by_str(const std::string type_str)
	{
		auto iter = RegisteredNodeType.find(type_str);
		if (iter == RegisteredNodeType.end())
		{
			return {};
		}
		else
		{
			return {iter->second->type};
		}
	}
    
    static void deregister_all_node_types()
    {
        for (const auto& [_, ptr] : RegisteredNodeType)
        {
            delete ptr;
        }
    }

private:
	static std::unordered_map<std::string, node<model_datatype> *> RegisteredNodeType;

protected:
	static void _registerNodeType(const std::string &name, node<model_datatype> *node)
	{
		auto iter = RegisteredNodeType.find(name);
		if (iter == RegisteredNodeType.end())
		{
			RegisteredNodeType.emplace(name, node);
		}
		else
		{
			delete node;
		}
	}
};

template<typename model_datatype> std::unordered_map<std::string, node<model_datatype> *> node<model_datatype>::RegisteredNodeType;

template<typename model_datatype>
class normal_node : public node<model_datatype>
{
public:
	normal_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = normal;
	};
	
	static std::string type_name()
	{
		return "normal";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new normal_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new normal_node(_name, buf_size);
	}
 
	void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
	{
        if (!this->enable) return;
        this->solver->train(data, label, display);
	}
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
	}
};

template<typename model_datatype>
class observer_node : public node<model_datatype>
{
public:
	observer_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = observer;
	};
	
	static std::string type_name()
	{
		return "observer";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new observer_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new observer_node(_name, buf_size);
	}
	
	void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
	{
        if (!this->enable) return;

	}
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
		return {};
	}
};


template<typename model_datatype>
class no_training_node : public node<model_datatype>
{
public:
    no_training_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
    {
        this->type = no_training;
    };
    
    static std::string type_name()
    {
        return "no_training";
    }
    
    static void registerNodeType()
    {
        node<model_datatype>::_registerNodeType(type_name(), new no_training_node("template", 0));
    }
    
    node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
    {
        return new no_training_node(_name, buf_size);
    }
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;

    }
    
    std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
    {
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
    }
};

template<typename model_datatype>
class pontificator_node : public node<model_datatype>
{
private:
    std::optional<Ml::caffe_parameter_net<model_datatype>> parameter;

public:
    pontificator_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
    {
        this->type = pontificator;
        this->enable_averaging = false;
    };

    static std::string type_name()
    {
        return "pontificator";
    }

    static void registerNodeType()
    {
        node<model_datatype>::_registerNodeType(type_name(), new pontificator_node("template", 0));
    }

    node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
    {
        return new pontificator_node(_name, buf_size);
    }

    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;
        if (!parameter)
            parameter = {this->solver->get_parameter()};
        this->solver->set_parameter(*parameter);
    }

    std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
    {
        if (!this->enable) return {};
        return parameter;
    }
};

template<typename model_datatype>
class malicious_model_poisoning_random_model_node : public node<model_datatype>
{
public:
	malicious_model_poisoning_random_model_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = malicious_model_poisoning_random_model;
	};
	
	static std::string type_name()
	{
		return "malicious_model_poisoning_random_model";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new malicious_model_poisoning_random_model_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new malicious_model_poisoning_random_model_node(_name, buf_size);
	}
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;
        this->solver->train(data, label, display);
    }
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
		Ml::caffe_parameter_net<model_datatype> output = this->solver->get_parameter();
		output.random(0,0.001);

		return {output};
	}
};

template<typename model_datatype>
class malicious_model_poisoning_random_model_by_turn_node : public node<model_datatype>
{
	/**
	 * Strategy: use trained model and random model by turn
	 **/
public:
	
	malicious_model_poisoning_random_model_by_turn_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = malicious_model_poisoning_random_model_by_turn;
		turn = 0;
	};
	
	static std::string type_name()
	{
		return "malicious_model_poisoning_random_model_by_turn";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new malicious_model_poisoning_random_model_by_turn_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new malicious_model_poisoning_random_model_by_turn_node(_name, buf_size);
	}
	
	int turn;
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;
        this->solver->train(data, label, display);
    }
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
		Ml::caffe_parameter_net<model_datatype> output;
		if (turn == 0)
		{
			output = this->solver->get_parameter();
		}
		else
		{
			output = this->solver->get_parameter();
			output.random(0,0.001);
		}
		turn = (turn + 1) % 2;
		return {output};
	}
	
};

template<typename model_datatype>
class malicious_model_poisoning_random_model_biased_0_1_node : public node<model_datatype>
{
public:
	malicious_model_poisoning_random_model_biased_0_1_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = malicious_model_poisoning_random_model_biased_0_1;
	};
	
	static std::string type_name()
	{
		return "malicious_model_poisoning_random_model_biased_0_1";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new malicious_model_poisoning_random_model_biased_0_1_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new malicious_model_poisoning_random_model_biased_0_1_node(_name, buf_size);
	}
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;
        this->solver->train(data, label, display);
    }
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
		Ml::caffe_parameter_net<model_datatype> output = this->solver->get_parameter();
		auto factor = output;
		factor.random(0, 0.1);
		output = output - factor;
		return {output};
	}
};

template<typename model_datatype>
class malicious_duplication_attack_node : public node<model_datatype>
{
public:
	malicious_duplication_attack_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = malicious_duplication_attack;
	};
	
	static std::string type_name()
	{
		return "malicious_duplication_attack";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new malicious_duplication_attack_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new malicious_duplication_attack_node(_name, buf_size);
	}
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;

    }
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
	}
};

template<typename model_datatype>
class malicious_data_poisoning_shuffle_label_node : public node<model_datatype>
{
public:
	malicious_data_poisoning_shuffle_label_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = malicious_data_poisoning_shuffle_label;
	};
	
	static std::string type_name()
	{
		return "malicious_data_poisoning_shuffle_label";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new malicious_data_poisoning_shuffle_label_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new malicious_data_poisoning_shuffle_label_node(_name, buf_size);
	}
	
	void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
	{
        if (!this->enable) return;

		std::vector<const Ml::tensor_blob_like<model_datatype>*> label_duplicate;
        label_duplicate.reserve(label.size());
        
        static std::default_random_engine generator;
        std::uniform_int_distribution<int> dist(0, 9);
        for (const auto& single_Label: label)
        {
            auto* temp = new Ml::tensor_blob_like<model_datatype>();
            *temp = *single_Label;
            //shuffle_label
            auto& labels = temp->getData();
            for (auto&& value: labels)
            {
                value = dist(generator);
            }
            label_duplicate.push_back(temp);
        }
        
        this->solver->train(data, label_duplicate, display);
        
        for (const auto& single_label: label_duplicate)
        {
            delete single_label;
        }
	}
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
	}
};

template<typename model_datatype>
class malicious_data_poisoning_shuffle_label_biased_1_node : public node<model_datatype>
{
public:
	malicious_data_poisoning_shuffle_label_biased_1_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = malicious_data_poisoning_shuffle_label_biased_1;
	};
	
	static std::string type_name()
	{
		return "malicious_data_poisoning_shuffle_label_biased_1";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new malicious_data_poisoning_shuffle_label_biased_1_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new malicious_data_poisoning_shuffle_label_biased_1_node(_name, buf_size);
	}
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
	{
        if (!this->enable) return;

        std::vector<const Ml::tensor_blob_like<model_datatype>*> label_duplicate;
        label_duplicate.reserve(label.size());
        for (const auto& single_label: label)
        {
            auto* temp = new Ml::tensor_blob_like<model_datatype>();
            *temp = *single_label;
            //shuffle_label_biased_1
            auto& labels = temp->getData();
            for (auto&& malicious_label : labels)
            {
                malicious_label++;
                if (malicious_label == 10) malicious_label = 0;
            }
            
            label_duplicate.push_back(temp);
        }
        
        this->solver->train(data, label_duplicate, display);
        
        for (const auto& single_label: label_duplicate)
        {
            delete single_label;
        }
	}
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
	}
};

template<typename model_datatype>
class malicious_data_poisoning_random_data_node : public node<model_datatype>
{
public:
	malicious_data_poisoning_random_data_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
	{
		this->type = malicious_data_poisoning_random_data;
	};
	
	static std::string type_name()
	{
		return "malicious_data_poisoning_random_data";
	}
	
	static void registerNodeType()
	{
		node<model_datatype>::_registerNodeType(type_name(), new malicious_data_poisoning_random_data_node("template", 0));
	}
	
	node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
	{
		return new malicious_data_poisoning_random_data_node(_name, buf_size);
	}
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
	{
        if (!this->enable) return;

        std::vector<const Ml::tensor_blob_like<model_datatype>*> data_duplicate;
        data_duplicate.reserve(data.size());
        for (const auto& single_data: data)
        {
            auto* temp = new Ml::tensor_blob_like<model_datatype>();
            *temp = *single_data;
            temp->random(0.0, 1.0); //data poisoning
            data_duplicate.push_back(temp);
        }
        
        this->solver->train(data_duplicate, label, display);
        
        for (const auto& single_data: data_duplicate)
        {
            delete single_data;
        }
	}
	
	std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
	{
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
	}
};

template<typename model_datatype>
class normal_node_label_0_4 : public node<model_datatype>
{
public:
    normal_node_label_0_4(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
    {
        this->type = normal_label_0_4;
    };

    static std::string type_name()
    {
        return "normal_node_label_0_4";
    }

    static void registerNodeType()
    {
        node<model_datatype>::_registerNodeType(type_name(), new normal_node_label_0_4("template", 0));
    }

    node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
    {
        return new normal_node_label_0_4(_name, buf_size);
    }

    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;
        this->solver->train(data, label, display);
    }

    std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
    {
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
    }
};

template<typename model_datatype>
class normal_node_label_5_9 : public node<model_datatype>
{
public:
    normal_node_label_5_9(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
    {
        this->type = normal_label_5_9;
    };

    static std::string type_name()
    {
        return "normal_node_label_5_9";
    }

    static void registerNodeType()
    {
        node<model_datatype>::_registerNodeType(type_name(), new normal_node_label_5_9("template", 0));
    }

    node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
    {
        return new normal_node_label_5_9(_name, buf_size);
    }

    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;
        this->solver->train(data, label, display);
    }

    std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
    {
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
    }
};

template<typename model_datatype>
class federated_learning_server_node : public node<model_datatype>
{
public:
    federated_learning_server_node(std::string _name, size_t buf_size) : node<model_datatype>(_name, buf_size)
    {
        this->type = federated_learning_server;
    };

    static std::string type_name()
    {
        return "federated_learning_server";
    }

    static void registerNodeType()
    {
        node<model_datatype>::_registerNodeType(type_name(), new federated_learning_server_node("template", 0));
    }

    node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
    {
        return new federated_learning_server_node(_name, buf_size);
    }

    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;


    }

    std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
    {
        if (!this->enable) return {};
        return {this->solver->get_parameter()};
    }
};

template<typename model_datatype>
class normal_node_reduced_sending : public node<model_datatype>
{
private:
    int reduce_factor;
    int send_counter;
    
public:
    normal_node_reduced_sending(std::string _name, size_t buf_size, int reduce_factor) : node<model_datatype>(_name, buf_size, {std::to_string(reduce_factor)})
    {
        this->type = normal_reduced_sending;
        this->send_counter = 0;
        this->reduce_factor = reduce_factor;
    };
    
    static std::string type_name()
    {
        return "normal_reduced_sending";
    }
    
    static void registerNodeType()
    {
        node<model_datatype>::_registerNodeType(type_name(), new normal_node_reduced_sending("template", 0, 0));
    }
    
    node<model_datatype> *new_node(std::string _name, size_t buf_size, std::optional<std::string> arg) override
    {
        LOG_ASSERT(arg.has_value());
        int arg_reduce_factor = std::stoi(*arg);
        return new normal_node_reduced_sending(_name, buf_size, arg_reduce_factor);
    }
    
    void train_model(const std::vector<const Ml::tensor_blob_like<model_datatype>*> &data, const std::vector<const Ml::tensor_blob_like<model_datatype>*> &label, bool display) override
    {
        if (!this->enable) return;
        this->solver->train(data, label, display);
    }
    
    std::optional<Ml::caffe_parameter_net<model_datatype>> generate_model_sent() override
    {
        if (!this->enable) return {};
        
        send_counter++;
        if (send_counter == reduce_factor) {
            send_counter = 0;
            return {this->solver->get_parameter()};
        }
        return {};
    }
};


template<typename model_datatype>
static void register_node_types()
{
	//register node types
	normal_node<model_datatype>::registerNodeType();
	observer_node<model_datatype>::registerNodeType();
    no_training_node<model_datatype>::registerNodeType();
    pontificator_node<model_datatype>::registerNodeType();
	malicious_model_poisoning_random_model_node<model_datatype>::registerNodeType();
	malicious_model_poisoning_random_model_by_turn_node<model_datatype>::registerNodeType();
	malicious_model_poisoning_random_model_biased_0_1_node<model_datatype>::registerNodeType();
	malicious_duplication_attack_node<model_datatype>::registerNodeType();
	malicious_data_poisoning_shuffle_label_node<model_datatype>::registerNodeType();
	malicious_data_poisoning_shuffle_label_biased_1_node<model_datatype>::registerNodeType();
	malicious_data_poisoning_random_data_node<model_datatype>::registerNodeType();

    normal_node_label_0_4<model_datatype>::registerNodeType();
    normal_node_label_5_9<model_datatype>::registerNodeType();

    federated_learning_server_node<model_datatype>::registerNodeType();
    
    normal_node_reduced_sending<model_datatype>::registerNodeType();
}

#endif //DFL_NODE_HPP
