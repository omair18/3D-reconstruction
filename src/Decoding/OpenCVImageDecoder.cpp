#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>

#include "OpenCVImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

namespace Decoding
{

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    std::vector<char> buffer(data, data  + size);
    decodedData = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
}

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedImage)
{
    cv::Mat decodedFrame;
    Decode(data, size, decodedFrame);
    decodedImage.upload(decodedFrame);
}

void OpenCVImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{
    cv::cuda::GpuMat gpuImage;
    Decode(data, size, gpuImage);
    decodedImage.MoveFromGpuMat(gpuImage);
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

void OpenCVImageDecoder::AllocateBuffer(int width, int height, int channels)
{
    LOG_WARNING() << "OpenCV image decoder doesn't have buffer.";
}

}
