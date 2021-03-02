#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#include <cstddef>

namespace DataStructures
{

struct CUDAImage final
{
    size_t width_ = 0;
    size_t height_ = 0;
    size_t pitch_ = 0;
    size_t channels_ = 0;
    size_t elementSize_ = 0;
    unsigned char* gpuData_ = nullptr;
};

}

#endif // CUDA_IMAGE_H