#pragma once

#include <mutex>
#include <ml_layer.hpp>

template <typename model_datatype>
class opti_model_update {
public:
	opti_model_update() = default;
	
	virtual void add_model(const Ml::caffe_parameter_net<model_datatype>& model) = 0;
	
	virtual Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<Ml::tensor_blob_like<model_datatype>>& test_data, const std::vector<Ml::tensor_blob_like<model_datatype>>& test_label) = 0;
	
	virtual size_t get_model_count() = 0;
};

template <typename model_datatype>
class train_50_average_50 : public opti_model_update<model_datatype> {
public:
	train_50_average_50() {
		_model_count = 0;
		_is_first_model = true;
	}

	void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
		std::lock_guard guard(_lock);
		_model_count++;
		if (_is_first_model) {
			_is_first_model = false;
			_buffered_model = model;
			return;
		}
		else {
			_buffered_model = _buffered_model + model;
			return;
		}
	}
	
	Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<Ml::tensor_blob_like<model_datatype>>& test_data, const std::vector<Ml::tensor_blob_like<model_datatype>>& test_label) override {
		std::lock_guard guard(_lock);
		auto output = self_model * 0.5 + _buffered_model/_model_count * 0.5;
		_model_count = 0;
		_is_first_model = true;
		return output;
	}
	
	size_t get_model_count() override {
		return _model_count;
	}
	
private:
	std::mutex _lock;
	bool _is_first_model;
	Ml::caffe_parameter_net<model_datatype> _buffered_model;
	size_t _model_count;
};