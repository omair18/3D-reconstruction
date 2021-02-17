#ifndef OPENCV_IMAGE_DECODER_H
#define OPENCV_IMAGE_DECODER_H

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

#endif // OPENCV_IMAGE_DECODER_H