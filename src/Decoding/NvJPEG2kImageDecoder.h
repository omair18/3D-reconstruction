#ifndef NVJPEG2K_DECODER_H
#define NVJPEG2K_DECODER_H

#include <nvjpeg2k.h>
#include <npp.h>

#include "IImageDecoder.h"

namespace Decoding
{

class NvJPEG2kImageDecoder final : public IImageDecoder
{
public:

    explicit NvJPEG2kImageDecoder(cudaStream_t& cudaStream);

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Initialize() override;

    bool IsInitialized() override;

    ~NvJPEG2kImageDecoder() override;


private:
    void DecodeInternal(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& outputImage);

    void AllocateBuffer(int width, int height, int channels) override;

    void InitDecoder();

    cudaStream_t cudaStream_;
};

}

#endif // NVJPEG2K_DECODER_H
