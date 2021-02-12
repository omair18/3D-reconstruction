#pragma once
#include "IImageDecoder.h"

class OpenCVImageDecoder final : public IImageDecoder
{
public:
    OpenCVImageDecoder() = default;

    ~OpenCVImageDecoder() override = default;

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    bool IsInitialized() override;
};

