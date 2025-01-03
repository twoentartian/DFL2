#pragma once

#include "./node.hpp"

//return: train_data,train_label
template<typename model_datatype>
std::tuple<std::vector<const Ml::tensor_blob_like<model_datatype>*>, std::vector<const Ml::tensor_blob_like<model_datatype>*>>
get_dataset_by_node_type(Ml::data_converter<model_datatype> &dataset, const node<model_datatype> &target_node, int size, const std::vector<int> &ml_dataset_all_possible_labels, bool use_random_training_sample_sequence=true)
{
	Ml::tensor_blob_like<model_datatype> label;
	label.getShape() = {1};
	std::vector<const Ml::tensor_blob_like<model_datatype>*> train_data, train_label;

    //process special cases first
    if (target_node.type == node_type::normal_label_0_4 || target_node.type == node_type::normal_label_5_9) {
        int min=0,max=0;
        if (target_node.type == node_type::normal_label_0_4) {
            min = 0;
            max = 4;
        }
        if (target_node.type == node_type::normal_label_5_9) {
            min = 5;
            max = 9;
        }
        LOG_ASSERT(min!=max);
        static std::random_device dev;
        static std::mt19937 rng(dev());
        std::uniform_int_distribution<int> distribution(min, max);
        for (int i = 0; i < size; ++i)
        {
            int label_int = ml_dataset_all_possible_labels[distribution(rng)];
            label.getData() = {model_datatype(label_int)};
            auto[train_data_slice, train_label_slice] = dataset.get_random_data_by_label(label, 1);
            assert(!train_data_slice.empty() && !train_label_slice.empty());
            train_data.insert(train_data.end(), train_data_slice.begin(), train_data_slice.end());
            train_label.insert(train_label.end(), train_label_slice.begin(), train_label_slice.end());
        }
        return {train_data, train_label};
    }

	if (target_node.dataset_mode == dataset_mode_type::default_dataset)
	{
		//iid dataset
		std::tie(train_data, train_label) = dataset.get_random_data(size, use_random_training_sample_sequence, target_node.name);
	}
	else if (target_node.dataset_mode == dataset_mode_type::iid_dataset)
	{
        LOG_ASSERT(use_random_training_sample_sequence) << "use_random_training_sample_sequence cannot be false for iid_dataset";
		static std::random_device dev;
		static std::mt19937 rng(dev());
		std::uniform_int_distribution<int> distribution(0, int(ml_dataset_all_possible_labels.size()) - 1);
		for (int i = 0; i < size; ++i)
		{
			int label_int = ml_dataset_all_possible_labels[distribution(rng)];
			label.getData() = {model_datatype(label_int)};
			auto[train_data_slice, train_label_slice] = dataset.get_random_data_by_label(label, 1);
            assert(!train_data_slice.empty() && !train_label_slice.empty());
			train_data.insert(train_data.end(), train_data_slice.begin(), train_data_slice.end());
			train_label.insert(train_label.end(), train_label_slice.begin(), train_label_slice.end());
		}
	}
	else if (target_node.dataset_mode == dataset_mode_type::non_iid_dataset)
	{
        LOG_ASSERT(use_random_training_sample_sequence) << "use_random_training_sample_sequence cannot be false for non_iid_dataset";

		//non-iid dataset
		static std::random_device dev;
		static std::mt19937 rng(dev());
		
		Ml::non_iid_distribution<model_datatype> label_distribution;
		for (auto &target_label : ml_dataset_all_possible_labels)
		{
			label.getData() = {model_datatype(target_label)};
			auto iter = target_node.special_non_iid_distribution.find(target_label);
			if (iter != target_node.special_non_iid_distribution.end())
			{
				auto[dis_min, dis_max] = iter->second;
				if (dis_min == dis_max)
				{
					label_distribution.add_distribution(label, dis_min);
				}
				else
				{
					std::uniform_real_distribution<model_datatype> distribution(dis_min, dis_max);
					label_distribution.add_distribution(label, distribution(rng));
				}
			}
			else
			{
				LOG(ERROR) << "cannot find the desired label";
			}
		}
		std::tie(train_data, train_label) = dataset.get_random_non_iid_dataset(label_distribution, size);
	}
	return {train_data, train_label};
}

//return <max, min>
template<typename T>
std::tuple<T, T> find_max_min(T* data, size_t size)
{
	T max, min;
	max=*data;min=*data;
	for (int i = 0; i < size; ++i)
	{
		if (data[i] > max) max = data[i];
		if (data[i] < min) min = data[i];
	}
	return {max,min};
}
