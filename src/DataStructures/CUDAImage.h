#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#include <cstddef>

struct CUDAImage
{
    size_t width = 0;
    size_t height = 0;
    size_t pitch = 0;
    size_t channels = 0;

    unsigned char* gpuData = nullptr;
};


#endif // CUDA_IMAGE_H
