#include <reputation_sdk.hpp>

template<typename DType>
class reputation_implementation : public reputation_interface<DType>
{
public:
    void update_model(Ml::caffe_parameter_net<DType> &current_model, double self_accuracy, const std::vector<updated_model<DType>> &models, std::unordered_map<std::string, double> &reputation) override
    {

    }
};

EXPORT_REPUTATION_API