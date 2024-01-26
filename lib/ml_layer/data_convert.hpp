//
// Created by gzr on 01-05-21.
//
#pragma once
#include <random>
#include <sstream>
#include <vector>
#include <memory>
#include <fstream>
#include <random>
#include <unordered_map>
#include <glog/logging.h>

#include <boost_serialization_wrapper.hpp>
#include "tensor_blob_like.hpp"
#include "exception.hpp"
#include "util.hpp"

namespace Ml{
	template <typename DType>
	class non_iid_distribution
	{
	public:
		void add_distribution(const tensor_blob_like<DType>& label, float weight)
		{
			std::string label_str = serialize_wrap<boost::archive::binary_oarchive>(label).str();
			_distribution[label_str] = weight;
		}
		
		const std::unordered_map<std::string, float>& get() const
		{
			return _distribution;
		}
		
	private:
		std::unordered_map<std::string, float> _distribution;
	};
	
    template <typename DType>
    class data_converter
    {
    public:
        void load_dataset_mnist(const std::string &image_filename, const std::string &label_filename) {
            std::ifstream data_file(image_filename, std::ios::in | std::ios::binary);
            std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
            CHECK(data_file) << "Unable to open file " << image_filename;
            CHECK(label_file) << "Unable to open file " << label_filename;

            uint32_t magic;
            uint32_t num_items;
            uint32_t num_labels;
            uint32_t rows;
            uint32_t cols;
            data_file.read(reinterpret_cast<char *>(&magic), 4);
            magic = swap_endian(magic);
            CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
            label_file.read(reinterpret_cast<char *>(&magic), 4);
            magic = swap_endian(magic);
            CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
            data_file.read(reinterpret_cast<char *>(&num_items), 4);
            num_items = swap_endian(num_items);

            label_file.read(reinterpret_cast<char *>(&num_labels), 4);
            num_labels = swap_endian(num_labels);
            CHECK_EQ(num_items, num_labels);
            data_file.read(reinterpret_cast<char *>(&rows), 4);
            rows = swap_endian(rows);
            data_file.read(reinterpret_cast<char *>(&cols), 4);
            cols = swap_endian(cols);
            
	        _data.resize(num_items);
	        _label.resize(num_items);
	        
	        char temp_label;
	        auto* temp_pixels = new char[rows * cols];
	        for (int item_id = 0; item_id < num_items; ++item_id)
	        {
		       auto& current_data_blob = _data[item_id].getData();
		       auto& current_label_blob = _label[item_id].getData();
		       _data[item_id].getShape() = {1, static_cast<int>(rows), static_cast<int>(cols)};
		       _label[item_id].getShape() = {1};
		
		       data_file.read(temp_pixels, rows * cols);
		       label_file.read(&temp_label, 1);
		       current_data_blob.resize(rows * cols);
		       for (int i = 0; i < rows * cols; ++i)
		       {
			       current_data_blob[i] = float(uint8_t(temp_pixels[i]));
		       }
		       current_label_blob.resize(1);
		       current_label_blob[0] = temp_label;
	        }
	        delete[] temp_pixels;
	        
	        //generate _data_tensor_by_label
	        for (int index = 0; index < num_items; ++index)
	        {
	        	std::string key = _label[index].get_str();
                //add to data container
                {
                    auto key_iter = _data_tensor_by_label.find(key);
                    if (key_iter == _data_tensor_by_label.end())
                    {
                        _data_tensor_by_label[key].reserve(num_items / 5);
                    }
                    _data_tensor_by_label[key].push_back(_data[index]);
                }
                //add to label container
                {
                    auto key_iter = _label_tensor_by_label.find(key);
                    if (key_iter == _label_tensor_by_label.end())
                    {
                        _label_tensor_by_label[key] = _label[index];
                    }
                }
	        }
	        for(auto&& iter: _data_tensor_by_label)
	        {
		        iter.second.shrink_to_fit();
	        }

            //generate a fixed sequence for use_random_data==false
            std::vector<size_t> sequence;
            sequence.reserve(num_items);
            for (int i = 0; i < num_items; ++i) {
                sequence.push_back(i);
            }
            std::shuffle(sequence.begin(), sequence.end(), std::mt19937(_dev()));
            _data_fixed_sequence.reserve(sequence.size());
            _label_fixed_sequence.reserve(sequence.size());
            for (const auto& index : sequence) {
                _data_fixed_sequence.push_back(&_data[index]);
                _label_fixed_sequence.push_back(&_label[index]);
            }
        }

        const std::vector<tensor_blob_like<DType>>& get_data() const
        {
            return _data;
        }

        const std::vector<tensor_blob_like<DType>>& get_label() const
        {
            return _label;
        }
        
        const std::unordered_map<std::string, std::vector<tensor_blob_like<DType>>>& get_container_by_label() const
        {
            return _data_tensor_by_label;
        }

        //return: <data,label>
        std::tuple<std::vector<const tensor_blob_like<DType>*>, std::vector<const tensor_blob_like<DType>*>> get_random_data(size_t size, bool use_random_data = true, const std::string& node_name = "")
        {
            if (use_random_data)
	            return _get_random_data(size, _data, _label);
            else {
                LOG_ASSERT(!node_name.empty());
                auto iter = _current_index_for_node_in_fixed_sequence.find(node_name);
                if (iter == _current_index_for_node_in_fixed_sequence.end()) {
                    const auto result = _current_index_for_node_in_fixed_sequence.emplace(node_name, 0);
                    iter = result.first;
                }

                std::vector<const tensor_blob_like<DType>*> output_data, output_label;
                size_t counter = 0;
                while (counter < size) {
                    output_data.push_back(_data_fixed_sequence[iter->second]);
                    output_label.push_back(_label_fixed_sequence[iter->second]);
                    iter->second ++;
                    if (iter->second == _data_fixed_sequence.size()) iter->second = 0;
                    counter ++;
                }
                LOG(INFO) << node_name << " gets until " << iter->second;
                return {output_data, output_label};
            }
        }
	
	    //return: <data,label>
	    std::tuple<std::vector<const tensor_blob_like<DType>*>, std::vector<const tensor_blob_like<DType>*>> get_random_data_by_label(const tensor_blob_like<DType>& arg_label, size_t size)
	    {
        	//does not exist key
        	const std::string key_str = arg_label.get_str();
			auto iter = _data_tensor_by_label.find(key_str);
			if(iter == _data_tensor_by_label.end())
			{
				return {{},{}};
			}
            
            const auto _label_ptr_iter = _label_tensor_by_label.find(key_str);
            if(_label_ptr_iter == _label_tensor_by_label.end())
            {
                return {{},{}};
            }
			
		    std::vector<const tensor_blob_like<DType>*> data,label;
		    data.resize(size);label.resize(size);
		    for (int i = 0; i < size; ++i)
		    {
			    static std::mt19937 rng(_dev());
			    std::uniform_int_distribution<int> distribution(0, iter->second.size()-1);
			    int dice = distribution(rng);
			    data[i] = &iter->second[dice];
			    label[i] = &_label_ptr_iter->second;
		    }
		    return {data,label};
	    }

        void append_random_data_by_label(const tensor_blob_like<DType>& arg_label, size_t size, std::vector<const tensor_blob_like<DType>*>& data, std::vector<const tensor_blob_like<DType>*>& label)
        {
            //does not exist key
            const std::string key_str = arg_label.get_str();
            auto iter = _data_tensor_by_label.find(key_str);
            if(iter == _data_tensor_by_label.end())
            {
                return;
            }
            auto label_ptr_iter = _label_tensor_by_label.find(key_str);
            if (label_ptr_iter == _label_tensor_by_label.end())
            {
                return;
            }
            data.reserve(data.size() + size);label.reserve(label.size() + size);
            for (int i = 0; i < size; ++i)
            {
                static std::mt19937 rng(_dev());
                std::uniform_int_distribution<int> distribution(0, iter->second.size()-1);
                int dice = distribution(rng);
                data.push_back(&iter->second[dice]);
                label.push_back(&label_ptr_iter->second);
            }
        }
	
	    //please ensure the dataset is larger than the size*100 to ensure the best randomness.
	    //return: <data,label>
	    std::tuple<std::vector<const tensor_blob_like<DType>*>, std::vector<const tensor_blob_like<DType>*>> get_random_non_iid_dataset(const non_iid_distribution<DType>& distribution, size_t size)
	    {
            std::vector<const tensor_blob_like<DType>*> output_data, output_label;
            
        	auto& distribution_map = distribution.get();
            std::unordered_map<std::string, std::pair<float,float>> boundaries;
            float counter = 0.0;
            for (auto iter = distribution_map.begin(); iter != distribution_map.end() ; iter++)
            {
                boundaries[iter->first] = std::make_pair(counter, counter + iter->second);
                counter += iter->second;
            }
            
            std::uniform_real_distribution<float> float_distribution(0, counter);
            for (int i = 0; i < size; ++i)
            {
                static std::mt19937 rng(_dev());
                auto random_number = float_distribution(rng);
                for (auto iter = boundaries.begin(); iter != boundaries.end() ; iter++)
                {
                    if (iter->second.second > random_number && random_number >= iter->second.first)
                    {
                        tensor_blob_like<DType> label_blob;
                        label_blob = deserialize_wrap<boost::archive::binary_iarchive, tensor_blob_like<DType>>(iter->first);
                        auto [data, label] = get_random_data_by_label(label_blob, 1);
                        output_data.push_back(data[0]);
                        output_label.push_back(label[0]);
                        break;
                    }
                }
            }
		    return {output_data, output_label};
	    }
	
	    std::tuple<const std::vector<tensor_blob_like<DType>>&, const std::vector<tensor_blob_like<DType>>& > get_whole_dataset()
	    {
		    return {_data, _label};
	    }
	    
    private:
        std::vector<tensor_blob_like<DType>> _data;
        std::vector<tensor_blob_like<DType>> _label;
        std::random_device _dev;
	
        std::unordered_map<std::string, std::vector<tensor_blob_like<DType>>> _data_tensor_by_label;
        std::unordered_map<std::string, tensor_blob_like<DType>> _label_tensor_by_label;

        std::vector<const tensor_blob_like<DType>*> _data_fixed_sequence;
        std::vector<const tensor_blob_like<DType>*> _label_fixed_sequence;
        std::unordered_map<std::string, int> _current_index_for_node_in_fixed_sequence;
	
	    //return: <data,label>
	    std::tuple<std::vector<const tensor_blob_like<DType>*>, std::vector<const tensor_blob_like<DType>*>> _get_random_data(size_t size, const std::vector<tensor_blob_like<DType>>& data_pool, const std::vector<tensor_blob_like<DType>>& label_pool)
	    {
		    const size_t& total_size = data_pool.size();
		    std::vector<const tensor_blob_like<DType>*> data,label;
		    data.resize(size);label.resize(size);
		    for (int i = 0; i < size; ++i)
		    {
			    static std::mt19937 rng(_dev());
			    std::uniform_int_distribution<size_t> distribution(0,total_size-1);
                auto dice = distribution(rng);
			    data[i] = &data_pool[dice];
			    label[i] = &label_pool[dice];
		    }
		    return {data,label};
	    }
     
	    static uint32_t swap_endian(uint32_t val)
	    {
		    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
		    return (val << 16) | (val >> 16);
	    }
	    
    };
}