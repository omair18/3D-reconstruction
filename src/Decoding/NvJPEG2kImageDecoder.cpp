#include <cuda_runtime.h>

#include "NvJPEG2kImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

#define CHECK_CUDA(status)                                                  \
{                                                                           \
    if (status != cudaSuccess)                                              \
    {                                                                       \
        LOG_ERROR() << "CUDA Runtime error " << static_cast<int>(status)    \
        << " : " << cudaGetErrorName(status) << " - "                       \
        << cudaGetErrorString(status);                                      \
        return;                                                             \
    }                                                                       \
}

#define CHECK_NVJPEG2K(status)                                              \
{                                                                           \
    if (status != nvjpeg2kStatus_t::NVJPEG2K_STATUS_SUCCESS)                \
    {                                                                       \
        LOG_ERROR() << "NvJPEG2K error " << static_cast<int>(status);       \
        return;                                                             \
    }                                                                       \
}

namespace Decoding
{

NvJPEG2kImageDecoder::NvJPEG2kImageDecoder(cudaStream_t cudaStream) :
cudaStream_(cudaStream),
bufferSize_(0),
initialized_(false)
{

}

NvJPEG2kImageDecoder::~NvJPEG2kImageDecoder()
{
    CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(jpegStream_));
    CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(decodeState_));
    CHECK_NVJPEG2K(nvjpeg2kDestroy(handle_));
}

void NvJPEG2kImageDecoder::AllocateBuffer(int width, int height, int channels)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{

}

void NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{

}

void NvJPEG2kImageDecoder::Initialize()
{

}

bool NvJPEG2kImageDecoder::IsInitialized()
{
    return initialized_;
}

void NvJPEG2kImageDecoder::DecodeInternal(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage& outputImage)
{

}

void NvJPEG2kImageDecoder::InitDecoder()
{

}

}
