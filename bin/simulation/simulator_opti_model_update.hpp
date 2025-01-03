#pragma once

#include <mutex>
#include <sstream>
#include <map>
#include <ml_layer.hpp>

template <typename model_datatype>
class opti_model_update {
public:
	opti_model_update() = default;

    virtual std::shared_ptr<opti_model_update> create_shared() = 0;

    virtual std::string get_name() = 0;

	virtual void add_model(const Ml::caffe_parameter_net<model_datatype>& model) = 0;
	
	virtual Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) = 0;
	
	virtual size_t get_model_count() = 0;

    static std::optional<std::shared_ptr<opti_model_update>> create_update_algorithm_from_name(const std::string& name) {
        auto iter = _all_update_algorithms.find(name);
        if (iter == _all_update_algorithms.end()) {
            return {};
        }
        return {iter->second->create_shared()};
    }

private:
    static std::map<std::string, std::shared_ptr<opti_model_update>> _all_update_algorithms;

protected:
    static void _registerAlgorithm(std::shared_ptr<opti_model_update> algo)
    {
        const std::string& name = algo->get_name();
        auto iter = _all_update_algorithms.find(name);
        if (iter == _all_update_algorithms.end())
        {
            _all_update_algorithms.emplace(name, algo);
        }
    }
};
template<typename model_datatype> std::map<std::string, std::shared_ptr<opti_model_update<model_datatype>>> opti_model_update<model_datatype>::_all_update_algorithms;

namespace opti_model_update_util {
    template<typename model_datatype>
    static model_datatype calculate_mean(const std::vector<model_datatype>& data) {
        model_datatype sum = 0.0;
        for (const double& value : data) {
            sum += value;
        }
        return sum / static_cast<double>(data.size());
    }

    template<typename model_datatype>
    static model_datatype calculate_variance(const std::vector<model_datatype>& data) {
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

    template<typename model_datatype>
    static std::map<std::string, model_datatype> get_variance_for_model(const Ml::caffe_parameter_net<model_datatype>& model) {
        std::map<std::string, model_datatype> variance_per_layer;
        for (const Ml::caffe_parameter_layer<model_datatype>& layer : model.getLayers()) {
            const auto blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                variance_per_layer.emplace(layer.getName(), calculate_variance(blobs[0]->getData()));
            }
        }
        return variance_per_layer;
    }

    template<typename model_datatype>
    static void scale_variance(std::vector<model_datatype>& data_series, model_datatype target_variance, float ratio=1.0f, float self_layer_variance=NAN) {
        model_datatype meanD1 = calculate_mean(data_series);
    
        model_datatype varD1 = 0;
        if (isnan(self_layer_variance)) {
            varD1 = calculate_variance(data_series);
        }
        else {
            varD1 = self_layer_variance;
        }

        // Calculate the standard deviation of D1 and the target standard deviation (sqrt of v2)
        model_datatype stdD1 = std::sqrt(varD1);
        model_datatype targetStd = std::sqrt(target_variance);

        // Scale each data point in D1
        const float scale_factor = (targetStd/stdD1-1)*ratio + 1;
        std::transform(data_series.begin(), data_series.end(), data_series.begin(), [&](model_datatype value) {
            return (value - meanD1) * scale_factor + meanD1;
        });
    }
    
    template<typename model_datatype>
    static model_datatype calculate_norm_for_layer(std::vector<model_datatype>& data_series) {
        model_datatype output = 0;
        for (const auto& v : data_series) {
            output += v * v;
        }
        return std::sqrt(output);
    }
    
    template<typename model_datatype>
    static std::map<std::string, model_datatype> calculate_norm(const Ml::caffe_parameter_net<model_datatype>& model) {
        std::map<std::string, model_datatype> norm;
        for (const Ml::caffe_parameter_layer<model_datatype>& layer : model.getLayers()) {
            const auto blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                norm.emplace(layer.getName(), calculate_norm_for_layer(blobs[0]->getData()));
            }
        }
        return norm;
    }
    
    template<typename model_datatype>
    static Ml::caffe_parameter_net<model_datatype> calculate_angle_norm(const Ml::caffe_parameter_net<model_datatype>& model, const std::map<std::string, model_datatype>& norm) {
        Ml::caffe_parameter_net<model_datatype> output = model.deep_clone();
        for (const Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
            const auto blobs = layer.getBlob_p();
            const auto layer_name = layer.getName();
            if (blobs.empty()) continue;
            auto& layer_weights = blobs[0]->getData();
            auto layer_norm_iter = norm.find(layer_name);
            model_datatype layer_norm = layer_norm_iter->second;
            for (auto& v : layer_weights) {
                v = v / layer_norm;
            }
        }
        return output;
    }
}

template <typename model_datatype>
class train_50_average_50 : public opti_model_update<model_datatype> {
public:
	train_50_average_50() {
		_model_count = 0;
		_is_first_model = true;
	}

    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_50_average_50>();
    }

    std::string get_name() override {
        return "train_50_average_50";
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
	
	Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
		std::lock_guard guard(_lock);
		auto output = self_model * 0.5 + _buffered_model/_model_count * 0.5;
		_model_count = 0;
		_is_first_model = true;
		return output;
	}
	
	size_t get_model_count() override {
		return _model_count;
	}

    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_50_average_50>());
    }
	
private:
	std::mutex _lock;
	bool _is_first_model;
	Ml::caffe_parameter_net<model_datatype> _buffered_model;
	size_t _model_count;
};

template <typename model_datatype>
class train_50_average_50_fix_variance_auto : public opti_model_update<model_datatype> {
public:
    train_50_average_50_fix_variance_auto() {
        _model_count = 0;
        _is_first_model = true;
    }

    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_50_average_50_fix_variance_auto>();
    }

    std::string get_name() override {
        return "train_50_average_50_fix_variance_auto";
    }

    void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
        std::lock_guard guard(_lock);
        _model_count++;
        if (_is_first_model) {
            _is_first_model = false;
            _buffered_model = model;
        }
        else {
            _buffered_model = _buffered_model + model;
        }
        //add variance
        std::map<std::string, model_datatype> variance_per_layer = opti_model_update_util::get_variance_for_model(model);
        _variances.push_back(variance_per_layer);

        return;
    }

    Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
        std::lock_guard guard(_lock);
        Ml::caffe_parameter_net<model_datatype> output = self_model * 0.5 + _buffered_model/_model_count * 0.5;
        //modify variance
        std::map<std::string, model_datatype> self_variance = opti_model_update_util::get_variance_for_model(output);
        std::map<std::string, model_datatype> sum_variance;
        for (const auto& variance_of_a_model : _variances) {
            for (const auto& [layer_name, variance] : variance_of_a_model) {
                sum_variance[layer_name] += variance;
            }
        }
        for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
            const std::string& name = layer.getName();
            const auto& blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                auto self_layer_variance = self_variance[name];
                auto sum_layer_variance = sum_variance[name];
                auto target_variance = sum_layer_variance / _model_count;
//                scale_variance(blobs[0]->getData(), self_layer_variance + (average_variance[name] - self_layer_variance) *(variance_ratio_to_average_received_model_variance));
                //factor = 1.01 -> loss=NAN at tick 5030
                //factor = 0.99 -> no NAN until tick 10000

                opti_model_update_util::scale_variance(blobs[0]->getData(), (sum_layer_variance / _model_count)*1.00f);

                LOG(INFO) << info << ", layer " << layer.getName() << ", scale variance from " << self_layer_variance << " to " << target_variance << " -- " << sum_layer_variance << "(total variance)" << "/" << _model_count;
            }
        }

        std::map<std::string, model_datatype> self_variance_after_scaling = opti_model_update_util::get_variance_for_model(output);

        _model_count = 0;
        _variances.clear();
        _is_first_model = true;
        return output;
    }

    size_t get_model_count() override {
        return _model_count;
    }

    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_50_average_50_fix_variance_auto>());
    }

protected:
    std::mutex _lock;
    bool _is_first_model;
    Ml::caffe_parameter_net<model_datatype> _buffered_model;
    std::vector<std::map<std::string, model_datatype>> _variances;
    size_t _model_count;
};

template <typename model_datatype>
class train_50_average_50_fix_variance_self : public opti_model_update<model_datatype> {
public:
    train_50_average_50_fix_variance_self() {
        _model_count = 0;
        _is_first_model = true;
    }
    
    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_50_average_50_fix_variance_self>();
    }
    
    std::string get_name() override {
        return "train_50_average_50_fix_variance_self";
    }
    
    void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
        std::lock_guard guard(_lock);
        _model_count++;
        if (_is_first_model) {
            _is_first_model = false;
            _buffered_model = model;
        }
        else {
            _buffered_model = _buffered_model + model;
        }
        
        return;
    }
    
    Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
        std::lock_guard guard(_lock);
        std::map<std::string, model_datatype> self_old_variance = opti_model_update_util::get_variance_for_model(self_model);
        Ml::caffe_parameter_net<model_datatype> output = self_model * 0.5 + _buffered_model/_model_count * 0.5;
        //modify variance
        std::map<std::string, model_datatype> self_variance = opti_model_update_util::get_variance_for_model(output);
        for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
            const std::string& name = layer.getName();
            const std::string layer_type = layer.getType();
//            if (layer_type == "BatchNorm") continue;
            if (layer_type == "Scale") continue;

            const auto& blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                auto self_layer_variance = self_variance[name];
                auto target_variance = self_old_variance[name];
//                scale_variance(blobs[0]->getData(), self_layer_variance + (average_variance[name] - self_layer_variance) *(variance_ratio_to_average_received_model_variance));
                //factor = 1.01 -> loss=NAN at tick 5030
                //factor = 0.99 -> no NAN until tick 10000
                
                opti_model_update_util::scale_variance(blobs[0]->getData(), target_variance);
                
                LOG(INFO) << info << ", layer " << layer.getName() << ", scale variance from " << self_layer_variance << " to " << target_variance << " (self old variance)";
            }
        }
        
        std::map<std::string, model_datatype> self_variance_after_scaling = opti_model_update_util::get_variance_for_model(output);
        
        _model_count = 0;
        _is_first_model = true;
        return output;
    }
    
    size_t get_model_count() override {
        return _model_count;
    }
    
    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_50_average_50_fix_variance_self>());
    }

protected:
    std::mutex _lock;
    bool _is_first_model;
    Ml::caffe_parameter_net<model_datatype> _buffered_model;
    size_t _model_count;
};

template <typename model_datatype>
class train_50_average_50_fix_variance_auto_05 : public opti_model_update<model_datatype> {
public:
    train_50_average_50_fix_variance_auto_05() {
        _model_count = 0;
        _is_first_model = true;
    }

    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_50_average_50_fix_variance_auto_05>();
    }

    std::string get_name() override {
        return "train_50_average_50_fix_variance_auto_05";
    }

    void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
        std::lock_guard guard(_lock);
        _model_count++;
        if (_is_first_model) {
            _is_first_model = false;
            _buffered_model = model;
        }
        else {
            _buffered_model = _buffered_model + model;
        }
        //add variance
        std::map<std::string, model_datatype> variance_per_layer = opti_model_update_util::get_variance_for_model(model);
        _variances.push_back(variance_per_layer);

        return;
    }

    Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
        std::lock_guard guard(_lock);
        Ml::caffe_parameter_net<model_datatype> output = self_model * 0.5 + _buffered_model/_model_count * 0.5;
        //modify variance
        std::map<std::string, model_datatype> self_variance = opti_model_update_util::get_variance_for_model(output);
        std::map<std::string, model_datatype> sum_variance;
        for (const auto& variance_of_a_model : _variances) {
            for (const auto& [layer_name, variance] : variance_of_a_model) {
                sum_variance[layer_name] += variance;
            }
        }
        for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
            const std::string& name = layer.getName();
            const auto& blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                auto self_layer_variance = self_variance[name];
                auto sum_layer_variance = sum_variance[name];
                auto target_variance = sum_layer_variance / _model_count;
//                scale_variance(blobs[0]->getData(), self_layer_variance + (average_variance[name] - self_layer_variance) *(variance_ratio_to_average_received_model_variance));
                //factor = 1.01 -> loss=NAN at tick 5030
                //factor = 0.99 -> no NAN until tick 10000

                opti_model_update_util::scale_variance(blobs[0]->getData(), (sum_layer_variance / _model_count)*1.00f, 0.5);

                LOG(INFO) << info << ", layer " << layer.getName() << ", scale variance from " << self_layer_variance << " to " << target_variance << " -- " << sum_layer_variance << "(total variance)" << "/" << _model_count;
            }
        }

        std::map<std::string, model_datatype> self_variance_after_scaling = opti_model_update_util::get_variance_for_model(output);

        _model_count = 0;
        _variances.clear();
        _is_first_model = true;
        return output;
    }

    size_t get_model_count() override {
        return _model_count;
    }

    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_50_average_50_fix_variance_auto_05>());
    }

protected:
    std::mutex _lock;
    bool _is_first_model;
    Ml::caffe_parameter_net<model_datatype> _buffered_model;
    std::vector<std::map<std::string, model_datatype>> _variances;
    size_t _model_count;
};

template <typename model_datatype>
class train_50_average_50_fix_variance_auto_099 : public opti_model_update<model_datatype> {
public:
    train_50_average_50_fix_variance_auto_099() {
        _model_count = 0;
        _is_first_model = true;
    }

    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_50_average_50_fix_variance_auto_099>();
    }

    std::string get_name() override {
        return "train_50_average_50_fix_variance_auto_099";
    }

    void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
        std::lock_guard guard(_lock);
        _model_count++;
        if (_is_first_model) {
            _is_first_model = false;
            _buffered_model = model;
        }
        else {
            _buffered_model = _buffered_model + model;
        }
        //add variance
        std::map<std::string, model_datatype> variance_per_layer = opti_model_update_util::get_variance_for_model(model);
        _variances.push_back(variance_per_layer);

        return;
    }

    Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
        std::lock_guard guard(_lock);
        Ml::caffe_parameter_net<model_datatype> output = self_model * 0.5 + _buffered_model/_model_count * 0.5;
        //modify variance
        std::map<std::string, model_datatype> self_variance = opti_model_update_util::get_variance_for_model(output);
        std::map<std::string, model_datatype> sum_variance;
        for (const auto& variance_of_a_model : _variances) {
            for (const auto& [layer_name, variance] : variance_of_a_model) {
                sum_variance[layer_name] += variance;
            }
        }
        for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
            const std::string& name = layer.getName();
            const auto& blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                auto self_layer_variance = self_variance[name];
                auto sum_layer_variance = sum_variance[name];
                auto target_variance = sum_layer_variance / _model_count;
//                scale_variance(blobs[0]->getData(), self_layer_variance + (average_variance[name] - self_layer_variance) *(variance_ratio_to_average_received_model_variance));
                //factor = 1.01 -> loss=NAN at tick 5030
                //factor = 0.99 -> no NAN until tick 10000

                opti_model_update_util::scale_variance(blobs[0]->getData(), (sum_layer_variance / _model_count)*1.00f, 0.99);

                LOG(INFO) << info << ", layer " << layer.getName() << ", scale variance from " << self_layer_variance << " to " << target_variance << " -- " << sum_layer_variance << "(total variance)" << "/" << _model_count;
            }
        }

        std::map<std::string, model_datatype> self_variance_after_scaling = opti_model_update_util::get_variance_for_model(output);

        _model_count = 0;
        _variances.clear();
        _is_first_model = true;
        return output;
    }

    size_t get_model_count() override {
        return _model_count;
    }

    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_50_average_50_fix_variance_auto_099>());
    }

protected:
    std::mutex _lock;
    bool _is_first_model;
    Ml::caffe_parameter_net<model_datatype> _buffered_model;
    std::vector<std::map<std::string, model_datatype>> _variances;
    size_t _model_count;
};

template <typename model_datatype>
class train_100_average_0 : public opti_model_update<model_datatype> {
public:
	train_100_average_0() {
		_model_count = 0;
	}

    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_100_average_0>();
    }

    std::string get_name() override {
        return "train_100_average_0";
    }
	
	void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
	}
	
	Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
		return self_model;
	}
	
	size_t get_model_count() override {
		return _model_count;
	}

    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_100_average_0>());
    }

private:
	size_t _model_count;
};

template <typename model_datatype>
class train_0_average_100 : public opti_model_update<model_datatype> {
public:
    train_0_average_100() {
        _model_count = 0;
        _is_first_model = true;
    }

    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_0_average_100>();
    }

    std::string get_name() override {
        return "train_0_average_100";
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

    Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
        std::lock_guard guard(_lock);
        auto output = _buffered_model/_model_count;
        _model_count = 0;
        _is_first_model = true;
        return output;
    }

    size_t get_model_count() override {
        return _model_count;
    }

    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_0_average_100>());
    }

private:
    std::mutex _lock;
    bool _is_first_model;
    Ml::caffe_parameter_net<model_datatype> _buffered_model;
    size_t _model_count;
};

template <typename model_datatype>
class train_0_average_100_fix_variance_auto : public opti_model_update<model_datatype> {
public:
    train_0_average_100_fix_variance_auto() {
        _model_count = 0;
        _is_first_model = true;
    }

    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_0_average_100_fix_variance_auto>();
    }

    std::string get_name() override {
        return "train_0_average_100_fix_variance_auto";
    }

    void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
        std::lock_guard guard(_lock);
        _model_count++;
        if (_is_first_model) {
            _is_first_model = false;
            _buffered_model = model;
        }
        else {
            _buffered_model = _buffered_model + model;
        }
        //add variance
        std::map<std::string, model_datatype> variance_per_layer = opti_model_update_util::get_variance_for_model(model);
        _variances.push_back(variance_per_layer);

        return;
    }

    Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
        std::lock_guard guard(_lock);
        Ml::caffe_parameter_net<model_datatype> output = _buffered_model/_model_count;
        //modify variance
        std::map<std::string, model_datatype> self_variance = opti_model_update_util::get_variance_for_model(output);
        std::map<std::string, model_datatype> sum_variance;
        for (const auto& variance_of_a_model : _variances) {
            for (const auto& [layer_name, variance] : variance_of_a_model) {
                sum_variance[layer_name] += variance;
            }
        }
        for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
            const std::string& name = layer.getName();
            const auto& blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                auto self_layer_variance = self_variance[name];
                auto sum_layer_variance = sum_variance[name];
                auto target_variance = sum_layer_variance / _model_count;
//                scale_variance(blobs[0]->getData(), self_layer_variance + (average_variance[name] - self_layer_variance) *(variance_ratio_to_average_received_model_variance));
                //factor = 1.01 -> loss=NAN at tick 5030
                //factor = 0.99 -> no NAN until tick 10000

                opti_model_update_util::scale_variance(blobs[0]->getData(), (sum_layer_variance / _model_count)*1.00f);

                LOG(INFO) << info << ", layer " << layer.getName() << ", scale variance from " << self_layer_variance << " to " << target_variance << " -- " << sum_layer_variance << "(total variance)" << "/" << _model_count;
            }
        }

        std::map<std::string, model_datatype> self_variance_after_scaling = opti_model_update_util::get_variance_for_model(output);

        _model_count = 0;
        _variances.clear();
        _is_first_model = true;
        return output;
    }

    size_t get_model_count() override {
        return _model_count;
    }

    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_0_average_100_fix_variance_auto>());
    }

protected:
    std::mutex _lock;
    bool _is_first_model;
    Ml::caffe_parameter_net<model_datatype> _buffered_model;
    std::vector<std::map<std::string, model_datatype>> _variances;
    size_t _model_count;
};

template <typename model_datatype>
class train_average_standard : public opti_model_update<model_datatype> {
public:
    train_average_standard() {
        _model_count = 0;
        _is_first_model = true;
    }
    
    std::shared_ptr<opti_model_update<model_datatype>> create_shared() override {
        return std::make_shared<train_average_standard>();
    }
    
    std::string get_name() override {
        return "train_average_standard";
    }
    
    void add_model(const Ml::caffe_parameter_net<model_datatype>& model) override {
        std::lock_guard guard(_lock);
        _model_count++;
        if (_is_first_model) {
            _is_first_model = false;
            _buffered_model = model;
        }
        else {
            _buffered_model = _buffered_model + model;
        }
        //add variance
        std::map<std::string, model_datatype> variance_per_layer = opti_model_update_util::get_variance_for_model(model);
        _variances.push_back(variance_per_layer);
        
        return;
    }
    
    Ml::caffe_parameter_net<model_datatype> get_output_model(const Ml::caffe_parameter_net<model_datatype>& self_model, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_data, const std::vector<const Ml::tensor_blob_like<model_datatype>*>& test_label, const std::string& info, const std::map<std::string, std::string>& args = {}) override {
        std::lock_guard guard(_lock);
        Ml::caffe_parameter_net<model_datatype> output = _buffered_model/_model_count;
        
        //check args
        LOG_ASSERT(args.contains("beta"));
        float beta = std::stof(args.at("beta"));
        LOG_ASSERT(args.contains("treat_beta_as_step_length"));
        bool treat_beta_as_step_length;
        std::istringstream(args.at("treat_beta_as_step_length")) >> std::boolalpha >> treat_beta_as_step_length;
        LOG_ASSERT(args.contains("treat_beta_as_step_length_step_modifier_factor_of_difference"));
        float treat_beta_as_step_length_step_modifier_factor_of_difference = std::stof(args.at("treat_beta_as_step_length_step_modifier_factor_of_difference"));
        LOG_ASSERT(args.contains("variance_correction"));
        bool enable_variance_correction;
        std::istringstream(args.at("variance_correction")) >> std::boolalpha >> enable_variance_correction;
        LOG_ASSERT(args.contains("variance_correction_method"));
        std::string variance_correction_method = args.at("variance_correction_method");
        //variance_correction_method: self, others, follow_beta
        LOG_ASSERT(args.contains("skip_layers_variance_correction"));
        LOG_ASSERT(args.contains("skip_layers_averaging"));
        std::string skip_vc_layers_str = args.at("skip_layers_variance_correction");
        std::string skip_averaging_layer_str = args.at("skip_layers_averaging");
        std::set<std::string> skipped_vc_layers;
        std::set<std::string> skipped_averaging_layers;
        {
            //insert to skipped_vc_layers
            {
                std::vector<std::string> skipped_layers_vec = util::split(skip_vc_layers_str, ',');
                for (const auto &i: skipped_layers_vec)
                {
                    skipped_vc_layers.emplace(i);
                }
            }
            //check existence of layers
            {
                std::map<std::string, bool> check_exist;
                for (const auto& i : skipped_vc_layers)
                {
                    check_exist.emplace(i, false);
                }
                for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
                    const std::string layer_name = layer.getName();
                    const std::string layer_type = layer.getType();
                    {
                        auto iter = check_exist.find(layer_type);
                        if (iter != check_exist.end()) {
                            iter->second = true;
                        }
                    }
                    {
                        auto iter = check_exist.find(layer_name);
                        if (iter != check_exist.end()) {
                            iter->second = true;
                        }
                    }
                }
                for (const auto& [layer_name, exist] : check_exist)
                {
                    LOG_IF(FATAL, !exist) << layer_name << " specified in [skip_layers_variance_correction] does not appear in the model";
                }
            }
        }
        {
            //insert to skip_averaging_layer_str
            {
                std::vector<std::string> skipped_layers_vec = util::split(skip_averaging_layer_str, ',');
                for (const auto &i: skipped_layers_vec)
                {
                    skipped_averaging_layers.emplace(i);
                }
            }
            //check existence of layers
            {
                std::map<std::string, bool> check_exist;
                for (const auto& i : skipped_averaging_layers)
                {
                    check_exist.emplace(i, false);
                }
                for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
                    const std::string layer_name = layer.getName();
                    const std::string layer_type = layer.getType();
                    {
                        auto iter = check_exist.find(layer_type);
                        if (iter != check_exist.end()) {
                            iter->second = true;
                        }
                    }
                    {
                        auto iter = check_exist.find(layer_name);
                        if (iter != check_exist.end()) {
                            iter->second = true;
                        }
                    }
                }
                for (const auto& [layer_name, exist] : check_exist)
                {
                    LOG_IF(FATAL, !exist) << layer_name << " specified in [skip_layers_averaging] does not appear in the model";
                }
            }
        }
        
        //averaging
        if (treat_beta_as_step_length) {
            auto diff_model = output - self_model;
            std::map<std::string, model_datatype> diff_model_norm = opti_model_update_util::calculate_norm(diff_model);
            Ml::caffe_parameter_net<model_datatype> diff_model_angle = opti_model_update_util::calculate_angle_norm(diff_model, diff_model_norm);
            Ml::caffe_parameter_net<model_datatype> output_model = diff_model_angle;
            std::vector<Ml::caffe_parameter_layer<model_datatype>>& layers = diff_model_angle.getLayers();
            for (size_t i=0; i < layers.size(); i++) {
                const auto& layer = layers[i];
                const auto layer_name = layer.getName();
                const std::string layer_type = layer.getType();
                if (skipped_averaging_layers.contains(layer_type)) continue;
                if (skipped_averaging_layers.contains(layer_name)) continue;
                
                const auto layer_norm_iter = diff_model_norm.find(layer_name);
                if (layer_norm_iter == diff_model_norm.end()) {
                    continue;
                }
                const auto layer_norm = layer_norm_iter->second;
                auto step_from_diff_modifier = layer_norm * treat_beta_as_step_length_step_modifier_factor_of_difference;
                auto step = std::max(beta, step_from_diff_modifier);
                auto& output_layers = output_model.getLayers();
                output_layers[i] = layer * step;
            }
            
            output = self_model + output_model;
        }
        else {
            output = self_model * beta + output * (1-beta);
        }
        
        
        //modify variance
        std::map<std::string, model_datatype> self_variance = opti_model_update_util::get_variance_for_model(output);
        std::map<std::string, model_datatype> sum_variance;
        for (const auto& variance_of_a_model : _variances) {
            for (const auto& [layer_name, variance] : variance_of_a_model) {
                sum_variance[layer_name] += variance;
            }
        }
        for (Ml::caffe_parameter_layer<model_datatype>& layer : output.getLayers()) {
            const std::string& name = layer.getName();
            const std::string layer_type = layer.getType();
            if (skipped_vc_layers.contains(layer_type)) continue;
            if (skipped_vc_layers.contains(name)) continue;
            const auto& blobs = layer.getBlob_p();
            if (!blobs.empty()) {
                auto self_layer_variance = self_variance[name];
                auto sum_layer_variance = sum_variance[name];
                float target_variance = 0;
                if (variance_correction_method == "self") {
                    target_variance = self_layer_variance;
                }
                else if (variance_correction_method == "others") {
                    target_variance = sum_layer_variance / _model_count;
                }
                else if (variance_correction_method == "follow_beta") {
                    target_variance = sum_layer_variance / _model_count * (1-beta) + self_layer_variance * beta;
                }
                else {
                    LOG(FATAL) << "unknown variance_correction_method:" << variance_correction_method;
                }
                
                opti_model_update_util::scale_variance(blobs[0]->getData(), target_variance, 1.0f, float(self_layer_variance));
                
                LOG(INFO) << info << ", layer " << layer.getName() << ", scale variance from " << self_layer_variance << " to " << target_variance << " -- " << sum_layer_variance << "(total variance)" << "/" << _model_count;
            }
        }
        
        std::map<std::string, model_datatype> self_variance_after_scaling = opti_model_update_util::get_variance_for_model(output);
        
        _model_count = 0;
        _variances.clear();
        _is_first_model = true;
        return output;
    }
    
    size_t get_model_count() override {
        return _model_count;
    }
    
    static void register_algorithm() {
        opti_model_update<model_datatype>::_registerAlgorithm(std::make_shared<train_average_standard>());
    }

protected:
    std::mutex _lock;
    bool _is_first_model;
    Ml::caffe_parameter_net<model_datatype> _buffered_model;
    std::vector<std::map<std::string, model_datatype>> _variances;
    size_t _model_count;
};

template <typename model_datatype>
void register_model_updating_algorithms() {
    train_50_average_50<model_datatype>::register_algorithm();
    train_100_average_0<model_datatype>::register_algorithm();
    train_0_average_100<model_datatype>::register_algorithm();
    train_50_average_50_fix_variance_auto<model_datatype>::register_algorithm();
    train_0_average_100_fix_variance_auto<model_datatype>::register_algorithm();
    train_50_average_50_fix_variance_auto_05<model_datatype>::register_algorithm();
    train_50_average_50_fix_variance_auto_099<model_datatype>::register_algorithm();
    train_50_average_50_fix_variance_self<model_datatype>::register_algorithm();
    
    train_average_standard<model_datatype>::register_algorithm();
}


