#ifndef NVJPEG_HARDWARE_IMAGE_DECODER_H
#define NVJPEG_HARDWARE_IMAGE_DECODER_H

#include "NvJPEGImageDecoder.h"

namespace Decoding
{

class NvJPEGHardwareImageDecoder : public NvJPEGImageDecoder
{
    explicit NvJPEGHardwareImageDecoder(cudaStream_t& cudaStream);

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Initialize() override;

    bool IsInitialized() override;

    ~NvJPEGHardwareImageDecoder() override = default;

private:

    void AllocateBuffer(int width, int height, int channels) override;

    void InitDecoder();
};

}

#endif // NVJPEG_HARDWARE_IMAGE_DECODER_H
