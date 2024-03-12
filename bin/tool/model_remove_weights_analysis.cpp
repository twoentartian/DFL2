#include <measure_time.hpp>
#include <tmt.hpp>

#include <ml_layer/caffe.hpp>
#include <ml_layer/tensor_blob_like.hpp>
#include <ml_layer/data_convert.hpp>
#include <ml_layer/model_compress.hpp>

int main() {
    Ml::MlCaffeModel<float, caffe::SGDSolver> node;
    node.load_caffe_model("../../../dataset/MNIST/lenet_solver_memory.prototxt");
    
    const std::string train_x_path = "../../../dataset/MNIST/train-images.idx3-ubyte";
    const std::string train_y_path = "../../../dataset/MNIST/train-labels.idx1-ubyte";
    
    LOG(INFO) << "loading dataset";
    Ml::data_converter<float> dataset;
    dataset.load_dataset_mnist(train_x_path, train_y_path);
    LOG(INFO) << "loading dataset - done";
    
    for (int repeat = 0; repeat < 500; ++repeat)    //5000 ticks
    {
        auto [train_x, train_y] = dataset.get_random_data(64);
        node.train(train_x, train_y);
        auto [test_x, test_y] = dataset.get_random_data(100);
        auto accuracy = node.evaluation(test_x, test_y);
        std::cout << "repeat:" << repeat << "    accuracy:" << accuracy << std::endl;
    }
    
    auto model = node.get_parameter();
    
    std::vector<std::shared_ptr<Ml::MlCaffeModel<float, caffe::SGDSolver>>> nodes_for_evaluation;
    for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
        auto p = std::make_shared<Ml::MlCaffeModel<float, caffe::SGDSolver>>();
        p->load_caffe_model("../../../dataset/MNIST/lenet_solver_memory.prototxt");
        nodes_for_evaluation.push_back(p);
    }
    
    std::vector<std::pair<int, int>> simulation_args;
    for (int repeat = 0; repeat < 100; ++repeat) {
        for (int compress_ratio = 0; compress_ratio <= 100; compress_ratio ++) {
            simulation_args.emplace_back(repeat, compress_ratio);
        }
    }
    
    std::map<std::pair<int,int>, float> accuracy_results;
    std::mutex accuracy_results_lock;
    tmt::ParallelExecution([&dataset, model, &nodes_for_evaluation, &accuracy_results_lock, &accuracy_results](uint32_t index, uint32_t thread_index, const std::pair<int,int>& simulation_arg){
        auto model_copy = model.deep_clone();
        auto [repeat, compress_ratio] = simulation_arg;
        auto compress_ratio_f = float(compress_ratio) / 100.0f;
        auto compressed_model = Ml::model_compress::compress_by_random_sampling_get_model(model_copy, model_copy, compress_ratio_f, 0.0f);
        auto [test_x, test_y] = dataset.get_random_data(100);
        nodes_for_evaluation[thread_index]->set_parameter(compressed_model);
        auto accuracy = nodes_for_evaluation[thread_index]->evaluation(test_x, test_y);
        {
            std::lock_guard guard(accuracy_results_lock);
            accuracy_results[simulation_arg] = accuracy;
        }
    }, simulation_args.size(), simulation_args.data());
    
    //export map to csv
    std::ofstream csv_output("accuracy_of_compressed_models.csv", std::ios::binary);
    csv_output << "compress ratio" << "," << "repeat" << "," << "accuracy" << std::endl;
    for (const auto& [k,v] : accuracy_results) {
        const auto [repeat, compress] = k;
        csv_output << compress << "," << repeat << "," << v << std::endl;
    }
    csv_output.flush();
    csv_output.close();
    
    return 0;
}