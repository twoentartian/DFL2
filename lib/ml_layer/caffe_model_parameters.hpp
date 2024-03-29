#pragma once

#include <memory>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <caffe/solver.hpp>
#include <caffe/net.hpp>
#include <caffe/layer.hpp>
#include <ml_layer/ml_abs.hpp>

#include <util.hpp>
#include "tensor_blob_like.hpp"

#define EXCLUDE_BIAS 0

namespace Ml
{
    template <typename DType>
    class caffe_parameter_layer;
    template <typename DType>
    class caffe_parameter_net;
    class caffe_net_structure_layer;
    class caffe_net_structure;

    template <typename DType>
    class caffe_parameter_layer
    {
    public:
        caffe_parameter_layer() = default;
	
	    using DataType = DType;

        caffe_parameter_layer<DType> deep_clone() const {
            caffe_parameter_layer<DType> output;
            output._name = this->_name;
            output._type = this->_type;
            std::vector<boost::shared_ptr<tensor_blob_like<DType>>> blobs;
            for (const auto& self_blob : _blob_p) {
                boost::shared_ptr<tensor_blob_like<DType>> temp_p = boost::make_shared<tensor_blob_like<DType>>();
                *temp_p = self_blob->deep_clone();
                blobs.push_back(temp_p);
            }
            output._blob_p = blobs;
            return output;
        }

        void fromLayer(const caffe::Layer<DType>& layer)
        {

            _name = layer.layer_param().name();
            _type = layer.layer_param().type();
            auto* layer_without_const = const_cast<caffe::Layer<DType>*>(&layer);
            auto& blobs = layer_without_const->blobs();
            _blob_p.clear();
#if EXCLUDE_BIAS
            size_t blob_count = std::min<size_t>(blobs.size(), 1);
#else
            size_t blob_count = blobs.size();
#endif
            for (int i = 0; i < blob_count; ++i)
            {
                auto& blob_model = *(blobs[i]);//blobs 0 stores the model data, blobs 1 stores the bias
                boost::shared_ptr<Ml::tensor_blob_like<DType>> blob_p(new tensor_blob_like<DType>());
                blob_p->fromBlob(blob_model);
                _blob_p.push_back(blob_p);
            }
        }

        void toLayer(caffe::Layer<DType>& layer, bool reshape) const
        {
            if (layer.layer_param().name() != _name)
            {
                LOG(WARNING) << "layer name mismatch: " << layer.layer_param().name() << " != (this)" << _name;
            }
            if (layer.layer_param().type() != _type)
            {
                LOG(WARNING) << "layer type mismatch: " << layer.layer_param().type() << " != (this)" << _type;
            }

            auto& blobs = layer.blobs();
            //empty?
            if (_blob_p.empty())
            {
                blobs.clear();
                return;
            }
            //ensure the target blob is not empty.
            if (blobs.empty())
            {
                LOG(WARNING) << "layer "<< _name <<"'s blob is empty";
            }
            for (int i = 0; i < _blob_p.size(); ++i)
            {
                _blob_p[i]->toBlob(*(blobs[i]), reshape);
            }
        }

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & _name;
            ar & _type;
            ar & _blob_p;
        }
	
	    caffe_parameter_layer<DType> operator+(const caffe_parameter_layer<DType>& target) const
	    {
		    caffe_parameter_layer<DType> output = *this;
            for (int i = 0; i < this->_blob_p.size(); ++i)
            {
                output._blob_p[i].reset(new tensor_blob_like<DType>());
                *output._blob_p[i] = *(this->_blob_p[i]) + *target._blob_p[i];
            }
		    return output;
	    }
	
	    caffe_parameter_layer<DType> operator-(const caffe_parameter_layer<DType>& target) const
	    {
            caffe_parameter_layer<DType> output = *this;
            for (int i = 0; i < this->_blob_p.size(); ++i)
            {
                output._blob_p[i].reset(new tensor_blob_like<DType>());
                *output._blob_p[i] = *(this->_blob_p[i]) - *target._blob_p[i];
            }
            return output;
	    }
	
	    template<typename D>
	    caffe_parameter_layer<DType> operator/(const D& target) const
	    {
            caffe_parameter_layer<DType> output = *this;
            for (int i = 0; i < this->_blob_p.size(); ++i)
            {
                output._blob_p[i].reset(new tensor_blob_like<DType>());
                *output._blob_p[i] = *(this->_blob_p[i]) / target;
            }
            return output;
	    }
	
	    template<typename D>
	    caffe_parameter_layer<DType> operator*(const D& target) const
	    {
            caffe_parameter_layer<DType> output = *this;
            for (int i = 0; i < this->_blob_p.size(); ++i)
            {
                output._blob_p[i].reset(new tensor_blob_like<DType>());
                *output._blob_p[i] = *(this->_blob_p[i]) * target;
            }
            return output;
	    }
	
	    bool operator==(const caffe_parameter_layer<DType>& target) const
	    {
            bool same = true;
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                if (*(this->_blob_p[i]) != *(target._blob_p[i])) {
                    same = false;
                    break;
                }
            }
            return same;
	    }
	
	    bool operator!=(const caffe_parameter_layer<DType>& target) const
	    {
		    return this->operator!=(target);
	    }
	    
	    bool roughly_equal(const caffe_parameter_layer<DType>& target, DType diff_threshold) const
	    {
		    if (this->_blob_p->empty() != target._blob_p->empty()) return false;
            if (this->_blob_p->empty()) return true;
            bool roughly_equal = true;
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                if (!this->_blob_p->roughly_equal(*target._blob_p, diff_threshold)) {
                    roughly_equal = false;
                    break;
                }
            }
		    return roughly_equal;
	    }
	
	    [[nodiscard]] caffe_parameter_layer<DType> dot_divide(const caffe_parameter_layer<DType>& target) const
	    {
		    caffe_parameter_layer<DType> output = *this;
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                output._blob_p[i].reset(new tensor_blob_like<DType>());
                *output._blob_p[i] = this->_blob_p[i]->dot_divide(*target._blob_p[i]);
            }
		    return output;
	    }
	
	    [[nodiscard]] caffe_parameter_layer<DType> dot_product(const caffe_parameter_layer<DType>& target) const
	    {
		    caffe_parameter_layer<DType> output = *this;
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                output._blob_p[i].reset(new tensor_blob_like<DType>());
                *output._blob_p[i] = this->_blob_p[i]->dot_product(*target._blob_p[i]);
            }
		    return output;
	    }

        GENERATE_GET(_name, getName);
        GENERATE_GET(_type, getType);
        GENERATE_GET(_blob_p, getBlob_p);
	
	    void set_all(DType value)
	    {
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                _blob_p[i]->set_all(value);
            }
	    }
	    
	    void random(DType min, DType max)
	    {
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                _blob_p[i]->random(min, max);
            }
	    }
	
	    DType sum()
	    {
            DType output = 0;
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                output += this->_blob_p[i]->sum();
            }
            return output;
	    }
	    
	    void abs()
	    {
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                _blob_p[i]->abs();
            }
	    }
	
	    size_t size() const
	    {
            size_t output_size = 0;
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                output_size += this->_blob_p[i]->size();
            }
            return output_size;
	    }
	    
	    void patch_weight(const caffe_parameter_layer<DType>& patch, DType ignore = NAN)
	    {
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                _blob_p[i]->patch_weight(*patch._blob_p[i], ignore);
            }
	    }
	
	    void regulate_weights(DType min, DType max)
	    {
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                _blob_p[i]->regulate_weights(min, max);
            }
		}
	
	    size_t fix_nan()
	    {
            size_t ret = 0;
            for (int i = 0; i < this->_blob_p.size(); ++i) {
                ret += _blob_p[i]->fix_nan();
            }
            return ret;
		}
        
    private:
        friend class boost::serialization::access;

        std::string _name;
        std::string _type;
        std::vector<boost::shared_ptr<tensor_blob_like<DType>>> _blob_p;
    };

    template <typename DType>
    class caffe_parameter_net
    {
    public:
	    using DataType = DType;
    	
        caffe_parameter_net() = default;

        caffe_parameter_net<DType> deep_clone() const {
            caffe_parameter_net<DType> output;
            std::vector<caffe_parameter_layer<DType>> layers;
            for (const auto& self_layer : _layers) {
                layers.push_back(self_layer.deep_clone());
            }
            output._layers = layers;
            output._name = _name;
            return output;
        }

        void fromNet(const caffe::Net<DType>& net)
        {
            _name = net.name();
            _layers.clear();
            _layers.resize(net.layers().size());
            for (int i = 0; i < net.layers().size(); ++i)
            {
                _layers[i].fromLayer(*(net.layers()[i]));
            }
        }

        void toNet(caffe::Net<DType>& net, bool reshape = false) const
        {
            if (net.name() != _name)
            {
                LOG(WARNING) << "net name mismatch: " << net.name() << " != (this)" << _name;
            }
            if (net.layers().size() != _layers.size())
            {
                LOG(WARNING) << "net layer size mismatch: " << net.layers().size() << " != (this)" << _layers.size();
            }
            auto& layers_p = net.layers();

            for (int i = 0; i < _layers.size(); ++i)
            {
                auto& target_layer = *(layers_p[i]);
                _layers[i].toLayer(target_layer, reshape);
            }
        }

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & _name;
            ar & _layers;
        }
	
	    caffe_parameter_net<DType> operator+(const caffe_parameter_net<DType>& target) const
	    {
		    caffe_parameter_net<DType> output = *this;
		    for (int i = 0; i < output._layers.size(); ++i)
		    {
			    output._layers[i] = output._layers[i] + target._layers[i];
		    }
		    return output;
	    }
	
	    caffe_parameter_net<DType> operator-(const caffe_parameter_net<DType>& target) const
	    {
		    caffe_parameter_net<DType> output = *this;
		    for (int i = 0; i < output._layers.size(); ++i)
		    {
			    output._layers[i] = output._layers[i] - target._layers[i];
		    }
		    return output;
	    }
	
	    template<typename D>
	    caffe_parameter_net<DType> operator/(const D& target) const
	    {
		    caffe_parameter_net<DType> output = *this;
		    for (int i = 0; i < output._layers.size(); ++i)
		    {
			    output._layers[i] = output._layers[i] / target;
		    }
		    return output;
	    }
	
	    template<typename D>
	    caffe_parameter_net<DType> operator*(const D& target) const
	    {
		    caffe_parameter_net<DType> output = *this;
		    for (int i = 0; i < output._layers.size(); ++i)
		    {
			    output._layers[i] = output._layers[i] * target;
		    }
		    return output;
	    }
	
	    bool operator==(const caffe_parameter_net<DType>& target) const
	    {
		    if(target._layers.size() != this->_layers.size()) return false;
		    for (int i = 0; i < target._layers.size(); ++i)
		    {
			    if(target._layers[i] != this->_layers[i])
			    	return false;
		    }
		    return true;
	    }
	
	    bool operator!=(const caffe_parameter_net<DType>& target) const
	    {
        	return !(*this==target);
	    }
	
	    bool roughly_equal(const caffe_parameter_net<DType>& target, DType diff_threshold) const
	    {
		    if(target._layers.size() != this->_layers.size()) return false;
		    for (int i = 0; i < target._layers.size(); ++i)
		    {
			    if(!target._layers[i].roughly_equal(this->_layers[i], diff_threshold))
				    return false;
		    }
		    return true;
	    }
	
	    [[nodiscard]] caffe_parameter_net<DType> dot_divide(const caffe_parameter_net<DType>& target) const
	    {
		    caffe_parameter_net<DType> output = *this;
		    for (int i = 0; i < output._layers.size(); ++i)
		    {
			    output._layers[i] = output._layers[i].dot_divide(target._layers[i]);
		    }
		    return output;
	    }
	
	    [[nodiscard]] caffe_parameter_net<DType> dot_product(const caffe_parameter_net<DType>& target) const
	    {
		    caffe_parameter_net<DType> output = *this;
		    for (int i = 0; i < output._layers.size(); ++i)
		    {
			    output._layers[i] = output._layers[i].dot_product(target._layers[i]);
		    }
		    return output;
	    }
	
	    void set_all(DType value)
	    {
		    for (auto& single_layer: _layers)
		    {
			    single_layer.set_all(value);
		    }
	    }
	    
	    void random(DType min, DType max)
	    {
		    for (auto& single_layer: _layers)
		    {
			    single_layer.random(min, max);
		    }
	    }
	
	    DType sum()
	    {
        	DType output = 0;
		    for (auto& single_layer: _layers)
		    {
			    output += single_layer.sum();
		    }
		    return output;
	    }
	
	    void abs()
	    {
		    for (auto& single_layer: _layers)
		    {
			    single_layer.abs();
		    }
	    }
	
	    size_t size()
	    {
		    size_t output = 0;
		    for (auto& single_layer: _layers)
		    {
			    output += single_layer.size();
		    }
		    return output;
	    }
	
	    void patch_weight(const caffe_parameter_net<DType>& patch, DType ignore = NAN)
	    {
		    for (int i = 0; i < _layers.size(); ++i)
		    {
			    _layers[i].patch_weight(patch._layers[i], ignore);
		    }
	    }

        std::vector<DType> get_debug_values(size_t size=5) const
        {
            std::vector<DType> output;
            bool exit = false;
            for (const auto& layer : this->_layers) {
                for (const auto& blob : layer.getBlob_p()) {
                    for (const auto& v : blob->getData()) {
                        output.push_back(v);
                        if (output.size() == size) exit = true;
                        if (exit) break;
                    }
                    if (exit) break;
                }
                if (exit) break;
            }
            return output;
        }
		
		void regulate_weights(DType min, DType max)
		{
			for (int i = 0; i < this->_layers.size(); ++i)
			{
				this->_layers[i].regulate_weights(min, max);
			}
		}

        //return: nan found and fixed
		size_t fix_nan()
		{
            size_t ret = 0;
			for (int i = 0; i < this->_layers.size(); ++i)
			{
                ret += this->_layers[i].fix_nan();
			}
            return ret;
		}

        GENERATE_GET(_name, getName);
        GENERATE_GET(_layers, getLayers);
    private:
        friend class boost::serialization::access;

        std::string _name;
        std::vector<caffe_parameter_layer<DType>> _layers;
    };

    class caffe_net_structure_layer
    {
    public:
        caffe_net_structure_layer() = default;

        template<typename DType>
        void fromLayer(const caffe_parameter_layer<DType>& layer)
        {
            _name = layer.getName();
            _type = layer.getType();
            _shape.clear();
            const auto& blob_p = layer.getBlob_p();
            for (int i = 0; i < blob_p.size(); ++i)
            {
                _shape.push_back(blob_p[i]->getShape());
            }
        }

        template<typename DType>
        void fromLayer(const caffe::Layer<DType>& layer)
        {
            _name = layer.layer_param().name();
            _type = layer.layer_param().type();
            auto* layer_without_const = const_cast<caffe::Layer<DType>*>(&layer);
            auto& blobs = layer_without_const->blobs();
            _shape.clear();
            if (blobs.size() == 0)
            {
                return;
            }
            else
            {
                for (int i = 0; i < blobs.size(); ++i)
                {
                    auto& blob_model = *(blobs[i]);
                    _shape.push_back(blob_model->shape());
                }
            }
        }

        bool operator == (const caffe_net_structure_layer& target) const
        {
            if (target._name != this->_name)
            {
                return false;
            }
            if (target._shape != this->_shape)
            {
                return false;
            }
            return true;
        }

        bool operator != (const caffe_net_structure_layer& target) const
        {
            return !this->operator==(target);
        }

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & _name;
            ar & _type;
            ar & _shape;
        }

        GENERATE_GET(_name, getName);
        GENERATE_GET(_type, getType);
        GENERATE_GET(_shape, getShape);
    private:
        friend class boost::serialization::access;

        std::string _name;
        std::string _type;
        std::vector<std::vector<int>> _shape;
    };

    class caffe_net_structure
    {
    public:
        template<typename DType>
        caffe_net_structure(caffe_parameter_net<DType> net)
        {
            _name = net.getName();
            if constexpr (std::is_same_v<DType, float>)
            {
                _dataType = model_data_type::data_type_float;
            }
            if constexpr (std::is_same_v<DType, double>)
            {
                _dataType = model_data_type::data_type_double;
            }
            auto& net_layers = net.getLayers();
            _layers.resize(net_layers.size());
            for (int i = 0; i < _layers.size(); ++i)
            {
                _layers[i].fromLayer(net_layers[i]);
            }
        }

        bool operator == (const caffe_net_structure& target) const
        {
            if (_layers.size() != target._layers.size())
            {
                return false;
            }
            for (int i = 0; i < _layers.size(); ++i)
            {
                if (_layers[i] != target._layers[i])
                {
                    return false;
                }
            }
            return true;
        }

        bool operator != (const caffe_net_structure& target) const
        {
            return !this->operator==(target);
        }

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & _name;
            ar & _layers;
        }

        GENERATE_GET(_name, getName);
        GENERATE_GET(_layers, getLayers);
    private:
        friend class boost::serialization::access;

        std::string _name;
        model_data_type _dataType;
        std::vector<caffe_net_structure_layer> _layers;
    };
}