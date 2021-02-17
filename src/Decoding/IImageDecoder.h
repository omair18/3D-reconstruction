#ifndef INTERFACE_IMAGE_DECODER_H
#define INTERFACE_IMAGE_DECODER_H

namespace cv
{
    class Mat;

    namespace cuda
    {
        class GpuMat;
    }
}

class IImageDecoder
{
public:
    IImageDecoder() = default;

    virtual void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) = 0;

    virtual void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) = 0;

    virtual bool IsInitialized() = 0;

    virtual ~IImageDecoder() = default;
};

#endif // INTERFACE_IMAGE_DECODER_H