#ifndef OPENCV_IMAGE_DECODER_H
#define OPENCV_IMAGE_DECODER_H

#include "IImageDecoder.h"

namespace Decoding
{

class OpenCVImageDecoder final : public IImageDecoder
{
public:
    OpenCVImageDecoder() = default;

    ~OpenCVImageDecoder() override = default;

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Initialize() override;

    bool IsInitialized() override;
};

}

#endif // OPENCV_IMAGE_DECODER_H