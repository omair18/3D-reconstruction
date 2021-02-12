#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>

#include "OpenCVImageDecoder.h"

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

bool OpenCVImageDecoder::IsInitialized()
{
    return true;
}
