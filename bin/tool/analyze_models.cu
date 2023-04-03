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

class cuda_stream_manager
{
public:
    explicit cuda_stream_manager(size_t stream_count)
    {
        _cuda_streams.resize(stream_count);
        for (auto& stream: _cuda_streams)
        {
            checkCudaErrors(cudaStreamCreate(&stream));
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

static cuda_stream_manager static_cuda_stream_manager(8);

void sync_all_cuda_stream()
{
    static_cuda_stream_manager.all_stream_synchronize();
    checkCudaErrors(cudaStreamSynchronize(cudaStream_t(0)));
}

void allocate_and_copy_device_memory(float** temp_device_ptr, const float* host_data, size_t size)
{
#if ANALYZE_MODEL_USE_SINGLE_CUDA_STREAM
    checkCudaErrors(cudaMalloc((void **)temp_device_ptr, size));
    checkCudaErrors(cudaMemcpy(*temp_device_ptr, host_data, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaStreamSynchronize(cudaStream_t(0)));
#else
    checkCudaErrors(cudaMallocAsync((void **)temp_device_ptr, size, cudaStream_t(0)));
    checkCudaErrors(cudaMemcpyAsync(*temp_device_ptr, host_data, size, cudaMemcpyHostToDevice, cudaStream_t(0)));
    checkCudaErrors(cudaStreamSynchronize(cudaStream_t(0)));
#endif
}

std::vector<float> run_kernel(const std::vector<float>& weight_l, float* lhs_device_data, float* rhs_device_data)
{
#if not ANALYZE_MODEL_USE_SINGLE_CUDA_STREAM
    auto cuda_stream = static_cuda_stream_manager.get_cuda_stream();
#endif
    
    //allocate output region on gpu
    auto output_size_bit = weight_l.size() * sizeof(weight_l[0]);
    float* output_dev;

#if ANALYZE_MODEL_USE_SINGLE_CUDA_STREAM
    checkCudaErrors(cudaMalloc((void **)&output_dev, output_size_bit));
#else
    checkCudaErrors(cudaMallocAsync((void **)&output_dev, output_size_bit, cuda_stream));
#endif

    
    size_t data_size = weight_l.size();
    uint32_t block_count = data_size / BLOCK_SIZE + 1;
#if ANALYZE_MODEL_USE_SINGLE_CUDA_STREAM
    calculate_square_of_value<float><<<block_count, BLOCK_SIZE>>>(output_dev, lhs_device_data, rhs_device_data, weight_l.size());
#else
    calculate_square_of_value<float><<<block_count, BLOCK_SIZE, 0, cuda_stream>>>(output_dev, lhs_device_data, rhs_device_data, weight_l.size());
#endif

    getLastCudaError("fail to start kernel <<<calculate_square_of_value>>>");

    //copy back to host
    std::vector<float> host_buffer;
    host_buffer.resize(weight_l.size());
#if ANALYZE_MODEL_USE_SINGLE_CUDA_STREAM
    checkCudaErrors(cudaMemcpy(host_buffer.data(), output_dev, output_size_bit, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(output_dev));
    checkCudaErrors(cudaStreamSynchronize(cudaStream_t(0)));
#else
    checkCudaErrors(cudaMemcpyAsync(host_buffer.data(), output_dev, output_size_bit, cudaMemcpyDeviceToHost, cuda_stream));
    checkCudaErrors(cudaFreeAsync(output_dev, cuda_stream));
    checkCudaErrors(cudaStreamSynchronize(cuda_stream));
#endif
    
    return host_buffer;
}

void clear_gpu_memory(const std::map<std::string, float*>& node_layer_to_device_memory)
{
    for (const auto& [_, device_memory] : node_layer_to_device_memory)
    {
#if ANALYZE_MODEL_USE_SINGLE_CUDA_STREAM
        checkCudaErrors(cudaFree(device_memory));
#else
        checkCudaErrors(cudaFreeAsync(device_memory, cudaStream_t(0)));
#endif

    }
}

