#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <filesystem>
#include <thread>
#include <mutex>

#include <BS_thread_pool.hpp>

#include "cuda_util.hpp"

constexpr uint32_t BLOCK_SIZE = 1024;
template <typename T>
__global__ void calculate_square_of_value(T* output, T* input0, T* input1, uint32_t size)
{
    uint32_t block_x = blockIdx.x;
    uint32_t thread_x = threadIdx.x;
    uint32_t index = block_x * BLOCK_SIZE + thread_x;

    if (index >= size) return;
    auto diff = (input0[index] - input1[index]);
    output[index] = diff * diff;
}

std::vector<float> calculate_square_of_value_host(const std::vector<float>& input0, const std::vector<float>& input1)
{
    uint32_t output_size = input0.size();
    if (output_size > input1.size())
    {
        output_size = input1.size();
    }
    uint32_t input0_size = input0.size();
    uint32_t input1_size = input1.size();

    uint32_t block_count = output_size / BLOCK_SIZE;

    float* input0_dev;
    float* input1_dev;
    float* output_dev;
    cudaStream_t cuda_stream;
    checkCudaErrors(cudaStreamCreate(&cuda_stream));
    checkCudaErrors(cudaMalloc((void **)&input0_dev, input0_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&input1_dev, input1_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&output_dev, output_size * sizeof(float)));
    checkCudaErrors(cudaMemcpy(input0_dev, input0.data(), input0_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input1_dev, input1.data(), input1_size * sizeof(float), cudaMemcpyHostToDevice));

    calculate_square_of_value<float><<<block_count, BLOCK_SIZE, 0, cuda_stream>>>(output_dev, input0_dev, input1_dev, output_size);
    getLastCudaError("fail to start kernel <<<calculate_square_of_value>>>");
    checkCudaErrors(cudaStreamSynchronize(cuda_stream));

    std::vector<float> output;
    output.resize(output_size);
    checkCudaErrors(cudaMemcpy(output.data(), output_dev, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    return output;
}

class cuda_stream_manager
{
public:
    explicit cuda_stream_manager(size_t stream_count)
    {
        _cuda_streams.resize(stream_count);
        for (auto& stream: _cuda_streams)
        {
            cudaStreamCreate(&stream);
        }
    }
    
    cudaStream_t get_cuda_stream()
    {
        stream_index++;
        if (stream_index >= _cuda_streams.size()) stream_index = 0;
        return _cuda_streams[stream_index];
    }
    
    void all_stream_synchronize()
    {
        for (auto& stream: _cuda_streams)
        {
            checkCudaErrors(cudaStreamSynchronize(stream));
        }
    }
    
    size_t get_manager_stream_count()
    {
        return _cuda_streams.size();
    }
    
private:
    int stream_index = 0;
    std::vector<cudaStream_t> _cuda_streams;
};

static cuda_stream_manager static_cuda_stream_manager(16);

std::map<std::pair<std::string, std::string>, std::map<std::string, float>> calculate_model_distance_of_each_model_pair_gpu_kernel(const std::map<std::string, std::map<std::string, std::vector<float>>>& node_layer_weight)
{
    std::map<std::pair<std::string, std::string>, std::map<std::string, float>> output;
    std::mutex output_lck;
    
    //allocate output
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end() ; ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
        {
            if (iter_r == iter_l) continue;
            for (const auto &[layer_name, weight_l]: iter_l->second)
            {
                auto node_pair = std::make_pair(iter_l->first, iter_r->first);
                output[node_pair][layer_name] = 0;
            }
        }
    }
    
    std::map<std::string, float*> node_layer_to_device_memory;
    
    //copy layer weight to GPU
    {
        auto cu_stream = static_cuda_stream_manager.get_cuda_stream();
        for (const auto& [node_name, layer_weight] : node_layer_weight)
        {
            for (const auto& [layer, weight] : layer_weight)
            {
                float* temp_device_ptr;
                checkCudaErrors(cudaMallocAsync((void **)&temp_device_ptr, weight.size() * sizeof(weight[0]), cu_stream));
                checkCudaErrors(cudaMemcpyAsync(temp_device_ptr, weight.data(), weight.size() * sizeof(weight[0]), cudaMemcpyHostToDevice, cu_stream));
                node_layer_to_device_memory.emplace(node_name+layer, temp_device_ptr);
            }
        }
        checkCudaErrors(cudaStreamSynchronize(cu_stream));
    }

    
    std::vector<std::thread> pools;
    for (auto iter_l = node_layer_weight.begin(); iter_l != node_layer_weight.end() ; ++iter_l)
    {
        for (auto iter_r = iter_l; iter_r != node_layer_weight.end(); ++iter_r)
        {
            if (iter_r == iter_l) continue;
            
            for (const auto& [layer_name, weight_l] : iter_l->second)
            {
                auto cuda_stream = static_cuda_stream_manager.get_cuda_stream();

                std::thread temp_thread([iter_l, iter_r, &node_layer_to_device_memory, &output, &cuda_stream, &output_lck, layer_name, weight_l](){
                    auto lhs_device_data_iter = node_layer_to_device_memory.find(iter_l->first + layer_name);
                    if (lhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                    float* lhs_device_data = lhs_device_data_iter->second;
                    
                    auto rhs_device_data_iter = node_layer_to_device_memory.find(iter_r->first + layer_name);
                    if (rhs_device_data_iter == node_layer_to_device_memory.end()) throw std::logic_error("logic_error");
                    float* rhs_device_data = rhs_device_data_iter->second;
                    
                    auto output_size_bit = weight_l.size() * sizeof(weight_l[0]);
                    
                    //allocate output region on gpu
                    float* output_dev;
                    checkCudaErrors(cudaMallocAsync((void **)&output_dev, output_size_bit, cuda_stream));
                    
                    size_t data_size = weight_l.size();
                    uint32_t block_count = data_size / BLOCK_SIZE + 1;
                    calculate_square_of_value<float><<<block_count, BLOCK_SIZE, 0, cuda_stream>>>(output_dev, lhs_device_data, rhs_device_data, weight_l.size());
                    getLastCudaError("fail to start kernel <<<calculate_square_of_value>>>");
                    
                    //copy back to host
                    std::vector<float> host_buffer;
                    host_buffer.resize(weight_l.size());
                    checkCudaErrors(cudaMemcpyAsync(host_buffer.data(), output_dev, output_size_bit, cudaMemcpyDeviceToHost, cuda_stream));
                    checkCudaErrors(cudaFreeAsync(output_dev, cuda_stream));
                    checkCudaErrors(cudaStreamSynchronize(cuda_stream));
                    
                    float v = 0;
                    for (const auto& i: host_buffer)
                    {
                        v += i;
                    }
                    auto node_pair = std::make_pair(iter_l->first, iter_r->first);
                    
                    {
                        std::lock_guard guard(output_lck);
                        output.at(node_pair).at(layer_name) = std::sqrt(v);
                    }
                });
                
                std::thread dummy;
                dummy.swap(temp_thread);
                pools.push_back(std::move(dummy));
            }
        }
    }
    for (auto& thread: pools)
    {
        thread.join();
    }
    
    static_cuda_stream_manager.all_stream_synchronize();
    
    //clear gpu memory
    {
        auto cu_stream = static_cuda_stream_manager.get_cuda_stream();
        for (const auto& [_, device_memory] : node_layer_to_device_memory)
        {
            checkCudaErrors(cudaFreeAsync(device_memory, cu_stream));
        }
        checkCudaErrors(cudaStreamSynchronize(cu_stream));
    }

    
    return output;
}
