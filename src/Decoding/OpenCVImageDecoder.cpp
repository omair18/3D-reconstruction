#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>

#include "OpenCVImageDecoder.h"

namespace Decoding
{

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    std::vector<char> buffer(data, data  + size);
    decodedData = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
}

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{
    cv::Mat decodedFrame;
    Decode(data, size, decodedFrame);
    decodedData.upload(decodedFrame);
}

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedImage, size_t outputWidth, size_t outputHeight)
{

}

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedImage, size_t outputWidth, size_t outputHeight)
{

}

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{

}

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage, size_t outputWidth, size_t outputHeight)
{

}

bool OpenCVImageDecoder::IsInitialized()
{
    return true;
}

}
