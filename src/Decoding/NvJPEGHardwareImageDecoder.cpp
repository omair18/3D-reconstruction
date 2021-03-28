#include "NvJPEGHardwareImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

#define CHECK_NVJPEG(status)                                                \
{                                                                           \
    if (status != NVJPEG_STATUS_SUCCESS)                                    \
    {                                                                       \
        LOG_ERROR() << "NvJPEG error " << static_cast<int>(status);         \
        return;                                                             \
    }                                                                       \
}

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

namespace Decoding
{

NvJPEGHardwareImageDecoder::NvJPEGHardwareImageDecoder(cudaStream_t cudaStream) :
NvJPEGImageDecoder(cudaStream)
{

}

void NvJPEGHardwareImageDecoder::AllocateBuffer(int width, int height, int channels)
{
    return NvJPEGImageDecoder::AllocateBuffer(width, height, channels);
}

void NvJPEGHardwareImageDecoder::InitDecoder()
{
    nvjpegStatus_t status = nvjpegCreateEx(nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE, nullptr,
                                           nullptr, NVJPEG_FLAGS_DEFAULT, &handle_);
    if (status != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
    {
        LOG_ERROR() << "Failed to create nvidia decoder.";
        return;
    }
    else
    {
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    }

    CHECK_NVJPEG(nvjpegDecoderCreate(handle_, nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE, &decoder_));
    CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, decoder_, &decoupledState_));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle_, nullptr, &pinnedBuffer_));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle_, nullptr, &deviceBuffer_));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &jpegStream_));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &decodeParams_));

    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(decoupledState_, deviceBuffer_));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoupledState_, pinnedBuffer_));
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decodeParams_, nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI));

    initialized_ = true;
}

void NvJPEGHardwareImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    return NvJPEGImageDecoder::Decode(data, size, decodedData);
}

void NvJPEGHardwareImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{
    return NvJPEGImageDecoder::Decode(data, size, decodedData);
}

void NvJPEGHardwareImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{
    return NvJPEGImageDecoder::Decode(data, size, decodedImage);
}

void NvJPEGHardwareImageDecoder::Initialize()
{
    InitDecoder();
}

bool NvJPEGHardwareImageDecoder::IsInitialized()
{
    return initialized_;
}


}