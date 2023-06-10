#include <ml_layer.hpp>
#include <memory_consumption.hpp>
#include "../../bin/simulation/simulator_opti_model_update.hpp"

int main() {
	train_50_average_50<float> buffer;
	
	Ml::MlCaffeModel<float,caffe::SGDSolver> model0;
	model0.load_caffe_model("../../../dataset/MNIST/lenet_solver_memory.prototxt");
    
    std::cout << get_memory_consumption_byte() / 1024 / 1024 << std::endl;
    
	for (int loop = 0; loop < 100; ++loop)
	{
		Ml::caffe_parameter_net<float> net0 = model0.get_parameter();
		for (int i = 0; i < 100; ++i)
		{
			buffer.add_model(net0);
		}
		
		auto output_net = buffer.get_output_model(net0, {}, {});
		
		if(!output_net.roughly_equal(net0, 0.001)) {
			printf("output_net.roughly_equal(net0, 0.001)) fail");
			return -1;
		}
		
		printf("finish %d\n", loop);
	}
    
    std::cout << get_memory_consumption_byte() / 1024 / 1024 << std::endl;
 
}