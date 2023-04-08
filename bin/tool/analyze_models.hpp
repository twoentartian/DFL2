#pragma once

#include <mutex>

std::mutex cout_mutex;

void calculate_distance(float *output, const std::vector<float> &data0, const std::vector<float> &data1)
{
    LOG_IF(FATAL, data0.size() != data1.size()) << "data sizes not equal";
    float v = 0;
    for (size_t index = 0; index < data0.size(); ++index)
    {
        auto t = data0[index] - data1[index];
        v += t * t;
    }
    *output = std::sqrt(v);
}

void calculate_distance_to_origin(float *output, const std::vector<float> &data0)
{
    float v = 0;
    for (size_t index = 0; index < data0.size(); ++index)
    {
        auto t = data0[index];
        v += t * t;
    }
    *output = std::sqrt(v);
}