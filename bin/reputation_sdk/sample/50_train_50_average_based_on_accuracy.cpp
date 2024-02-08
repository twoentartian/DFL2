#include <algorithm>
#include <reputation_sdk.hpp>
#include <clustering/hierarchical_clustering.hpp>

constexpr float SELF_MODEL_CONTRIBUTION = 0.5;

template<typename DType>
class reputation_implementation : public reputation_interface<DType>
{
public:
    void update_model(Ml::caffe_parameter_net<DType> &current_model, double self_accuracy, const std::vector<updated_model<DType>> &models, std::unordered_map<std::string, double> &reputation) override
    {
        if (models.empty()) return;

        ////////********////////    put the reputation logic here    ////////********////////
        ////////********  to update the model, directly change the "current_model"
        ////////********  to update the reputation, directly use "reputation[node_address]" to change

        Ml::caffe_parameter_net<DType> zero_parameter = models[0].model_parameter - models[0].model_parameter;
        zero_parameter.set_all(0);
        Ml::caffe_parameter_net<DType> average_part = zero_parameter;

        double total_accuracy = 0;
        total_accuracy += self_accuracy;//self model also counts
        for (const updated_model<DType>& model : models) {
            total_accuracy += model.accuracy;
        }
        average_part = average_part + current_model * (self_accuracy / total_accuracy * (1-SELF_MODEL_CONTRIBUTION));
        for (const updated_model<DType>& model : models) {
            average_part = average_part + model.model_parameter * (model.accuracy / total_accuracy * (1-SELF_MODEL_CONTRIBUTION));
        }
        current_model = current_model * SELF_MODEL_CONTRIBUTION + average_part;
    }

};

EXPORT_REPUTATION_API
