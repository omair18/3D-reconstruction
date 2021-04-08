#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>

#include "OpenCVImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

namespace Decoding
{

OpenCVImageDecoder::~OpenCVImageDecoder() noexcept(false)
{

}

bool OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    LOG_TRACE() << "Decoding image with OpenCV decoder";
    std::vector<char> buffer(data, data  + size);
    decodedData = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if(!decodedData.empty())
    {
        LOG_TRACE() << "Image was successfully decoded.";
        return true;
    }
    else
    {
        LOG_ERROR() << "Failed to decode image.";
        return false;
    }
}

bool OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedImage)
{
    cv::Mat decodedFrame;
    if(Decode(data, size, decodedFrame))
    {
        decodedImage.upload(decodedFrame);
        return true;
    }
    else
    {
        return false;
    }

}

bool OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{
    cv::cuda::GpuMat gpuImage;
    if(Decode(data, size, gpuImage))
    {
        decodedImage.MoveFromGpuMat(gpuImage);
        return true;
    }
    else
    {
        return false;
    }
}

bool OpenCVImageDecoder::IsInitialized()
{
    return true;
}

void OpenCVImageDecoder::Initialize()
{
    LOG_TRACE() << "Initializing OpenCV image decoder ...";
    LOG_TRACE() << "OpenCV decoder was successfully initialized_.";
}

}
