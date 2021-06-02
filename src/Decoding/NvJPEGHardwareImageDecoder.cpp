#include "NvJPEGHardwareImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

#define CHECK_NVJPEG(status)                                                \
{                                                                           \
    if (status != NVJPEG_STATUS_SUCCESS)                                    \
    {                                                                       \
        LOG_ERROR() << "NvJPEG error " << static_cast<int>(status);         \
        return false;                                                       \
    }                                                                       \
}

namespace Decoding
{

NvJPEGHardwareImageDecoder::NvJPEGHardwareImageDecoder(cudaStream_t cudaStream) :
NvJPEGImageDecoder(cudaStream)
{

}

NvJPEGHardwareImageDecoder::~NvJPEGHardwareImageDecoder() noexcept(false)
{
    if(initialized_)
    {
        nvjpegStatus_t status;
        if((status = nvjpegDecodeParamsDestroy(decodeParams_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG decoding params structure. Error: " << static_cast<int>(status);
        }

        if((status = nvjpegJpegStreamDestroy(jpegStream_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG decoding stream. Error: " << static_cast<int>(status);
        }

        if((status = nvjpegBufferPinnedDestroy(pinnedBuffer_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG pinned buffer. Error: " << static_cast<int>(status);
        }

        if((status = nvjpegBufferDeviceDestroy(deviceBuffer_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG device buffer. Error: " << static_cast<int>(status);
        }

        if((status = nvjpegJpegStateDestroy(decoupledState_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG decoding decoupled state structure. Error: " << static_cast<int>(status);
        }

        if((status = nvjpegDecoderDestroy(decoder_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG decoder. Error: " << static_cast<int>(status);
        }

        if((status = nvjpegJpegStateDestroy(state_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG decoding state structure. Error: " << static_cast<int>(status);
        }

        if((status = nvjpegDestroy(handle_)) != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG handle. Error: " << static_cast<int>(status);
        }

        cudaError_t cudaStatus;
        for (auto& channel : imageBuffer_.channel)
        {
            if (channel)
            {
                cudaStatus = cudaFree(channel);
                if(cudaStatus != cudaError_t::cudaSuccess)
                {
                    LOG_ERROR() << "Failed to release NvJPEG buffer. Error " << static_cast<int>(cudaStatus) << ": "
                    << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA error");
                }
            }
        }

        initialized_ = false;
    }
}

void NvJPEGHardwareImageDecoder::AllocateBuffer(int width, int height, int channels)
{
    return NvJPEGImageDecoder::AllocateBuffer(width, height, channels);
}

bool NvJPEGHardwareImageDecoder::InitDecoder()
{
    nvjpegStatus_t status = nvjpegCreateEx(nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE, nullptr,
                                           nullptr, NVJPEG_FLAGS_DEFAULT, &handle_);
    if (status != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
    {
        LOG_ERROR() << "Failed to create NvJPEG hardware decoder.";
        return false;
    }
    else
    {
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_))
    }

    CHECK_NVJPEG(nvjpegDecoderCreate(handle_, nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE, &decoder_))
    CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, decoder_, &decoupledState_))
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle_, nullptr, &pinnedBuffer_))
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle_, nullptr, &deviceBuffer_))
    CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &jpegStream_))
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &decodeParams_))

    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(decoupledState_, deviceBuffer_))
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(decoupledState_, pinnedBuffer_))
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(decodeParams_, nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI))

    initialized_ = true;
    return true;
}

bool NvJPEGHardwareImageDecoder::Decode(const unsigned char* data, unsigned long long int size, cv::Mat& decodedData)
{
    return NvJPEGImageDecoder::Decode(data, size, decodedData);
}

bool NvJPEGHardwareImageDecoder::Decode(const unsigned char* data, unsigned long long int size, cv::cuda::GpuMat& decodedData)
{
    return NvJPEGImageDecoder::Decode(data, size, decodedData);
}

bool NvJPEGHardwareImageDecoder::Decode(const unsigned char* data, unsigned long long int size, DataStructures::CUDAImage& decodedImage)
{
    return NvJPEGImageDecoder::Decode(data, size, decodedImage);
}

void NvJPEGHardwareImageDecoder::Initialize()
{
    LOG_TRACE() << "Initializing NvJPEG hardware image decoder ...";
    if(!initialized_)
    {
        if(InitDecoder())
        {
            LOG_TRACE() << "NvJPEG hardware image decoder was successfully initialized.";
        }
        else
        {
            LOG_ERROR() << "Failed to initialize NvJPEG hardware image decoder.";
        }
    }
    else
    {
        LOG_WARNING() << "NvJPEG hardware image decoder is already initialized.";
    }
}

bool NvJPEGHardwareImageDecoder::IsInitialized()
{
    return initialized_;
}

}