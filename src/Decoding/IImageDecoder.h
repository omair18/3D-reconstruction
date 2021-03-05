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

namespace DataStructures
{
    struct CUDAImage;
}

namespace Decoding
{

class IImageDecoder
{
public:
    IImageDecoder() = default;

    virtual ~IImageDecoder() = default;

    virtual void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage) = 0;

    virtual void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage, size_t outputWidth, size_t outputHeight) = 0;

    virtual void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage) = 0;

    virtual void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage, size_t outputWidth, size_t outputHeight) = 0;

    virtual void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) = 0;

    virtual void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage, size_t outputWidth, size_t outputHeight) = 0;

    virtual void Initialize() = 0;

    virtual bool IsInitialized() = 0;

protected:

    virtual void AllocateBuffer(int width, int height, int channels) = 0;

};

}

#endif // INTERFACE_IMAGE_DECODER_H