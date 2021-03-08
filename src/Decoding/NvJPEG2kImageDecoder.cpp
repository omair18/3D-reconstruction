#include <nppi_geometry_transforms.h>

#include "NvJPEG2kImageDecoder.h"

namespace Decoding
{

NvJPEG2kImageDecoder::NvJPEG2kImageDecoder(cudaStream_t &cudaStream) :
cudaStream_(cudaStream)
{

}

NvJPEG2kImageDecoder::~NvJPEG2kImageDecoder()
{

}

void NvJPEG2kImageDecoder::AllocateBuffer(int width, int height, int channels)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedImage, size_t outputWidth, size_t outputHeight)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedImage, size_t outputWidth, size_t outputHeight)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage, size_t outputWidth, size_t outputHeight)
{

}

void NvJPEG2kImageDecoder::Initialize()
{

}

bool NvJPEG2kImageDecoder::IsInitialized()
{
    return false;
}

void NvJPEG2kImageDecoder::DecodeInternal(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &outputImage)
{

}

void NvJPEG2kImageDecoder::InitDecoder()
{

}

}
