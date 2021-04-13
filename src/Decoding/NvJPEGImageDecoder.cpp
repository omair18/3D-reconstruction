#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>

#include "NvJPEGImageDecoder.h"
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

NvJPEGImageDecoder::NvJPEGImageDecoder(cudaStream_t cudaStream) :
initialized_(false),
cudaStream_(cudaStream),
bufferSize_(0)
{
    std::memset(&imageBuffer_, 0, sizeof(imageBuffer_));
}

NvJPEGImageDecoder::~NvJPEGImageDecoder() noexcept(false)
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
        for (auto &channel : imageBuffer_.channel)
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

bool NvJPEGImageDecoder::InitDecoder()
{
    nvjpegStatus_t status = nvjpegCreateEx(nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, nullptr,
                                               nullptr, NVJPEG_FLAGS_DEFAULT, &handle_);
    if (status != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
    {
        LOG_ERROR() << "Failed to create NvJPEG decoder.";
        return false;
    }
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_))
    CHECK_NVJPEG(nvjpegDecoderCreate(handle_, nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, &decoder_))
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

void NvJPEGImageDecoder::AllocateBuffer(int width, int height, int channels)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG image decoder must be initialized before allocating buffer.";
        return;
    }
    size_t imageSize = width * height * channels;
    cudaError_t cudaStatus;
    if(imageSize > bufferSize_)
    {
        LOG_TRACE() << "Allocating buffer for image " << width << "x" << height << " with " << channels << " channel(s)."
        << " Previous buffer size: " << bufferSize_;
        if (imageBuffer_.channel[0])
        {
            if((cudaStatus = cudaFree(imageBuffer_.channel[0])) != cudaError_t::cudaSuccess)
            {
                LOG_ERROR() << "Failed to release previous NvJPEG buffer. Error " << static_cast<int>(cudaStatus) << ": "
                << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
                throw std::runtime_error("CUDA error.");
            }
            else
            {
                LOG_TRACE() << "Previous NvJPEG buffer with size " << bufferSize_ << " bytes was successfully released.";
            }
        }
        if((cudaStatus = cudaMallocPitch(&imageBuffer_.channel[0], &imageBuffer_.pitch[0], width * channels, height)) != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to allocate pitched NvJPEG buffer. Error " << static_cast<int>(cudaStatus) << ": "
                        << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
            throw std::runtime_error("CUDA error.");
        }
        else
        {
            bufferSize_ = imageBuffer_.pitch[0] * height;
            LOG_TRACE() << "Allocated net pitched NvJPEG buffer. Buffer size: " << bufferSize_ << ", pitch: "
            << imageBuffer_.pitch[0];
        }
    }
    else
    {
        LOG_TRACE() << "Using existing buffer for image decoding.";
    }
}

bool NvJPEGImageDecoder::DecodeInternal(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage& image)
{
    unsigned int channels;
    unsigned int width;
    unsigned int height;

    CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, reinterpret_cast<const unsigned char *>(data), size, 0, 0, jpegStream_))

    CHECK_NVJPEG(nvjpegJpegStreamGetComponentsNum(jpegStream_, &channels))

    CHECK_NVJPEG(nvjpegJpegStreamGetComponentDimensions(jpegStream_, 0, &width, &height))

    LOG_TRACE() << "Decoding image " << width << "x" << height << " with " << channels << " channel(s) ...";

    AllocateBuffer(static_cast<int>(width),
                   static_cast<int>(height),
                   static_cast<int>(channels));

    CHECK_NVJPEG(nvjpegDecodeJpegHost(handle_, decoder_, decoupledState_, decodeParams_, jpegStream_))

    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, decoder_, decoupledState_, jpegStream_, cudaStream_))

    CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, decoder_, decoupledState_, &imageBuffer_, cudaStream_))

    DataStructures::CUDAImage decodedImage;
    decodedImage.gpuData_ = imageBuffer_.channel[0];
    decodedImage.width_ = width;
    decodedImage.height_ = height;
    decodedImage.channels_ = channels;
    decodedImage.pitch_ = imageBuffer_.pitch[0];
    decodedImage.elementType_ = DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U;
    decodedImage.pitchedAllocation_ = true;
    decodedImage.allocatedBytes_ = decodedImage.pitch_ * decodedImage.height_;

    image.CopyFromCUDAImageAsync(decodedImage, cudaStream_);

    decodedImage.gpuData_ = nullptr;

    return true;
}

bool NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG image decoder must be initialized before decoding.";
        return false;
    }
    DataStructures::CUDAImage image;
    if(DecodeInternal(data, size, image))
    {
        LOG_TRACE() << "Image was successfully decoded with NvJPEG decoder.";
        image.MoveToCvMatAsync(decodedData, cudaStream_);
        return true;
    }
    else
    {
        LOG_ERROR() << "Failed to decode image with NvJPEG decoder.";
        return false;
    }

}

bool NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG image decoder must be initialized before decoding.";
        return false;
    }
    DataStructures::CUDAImage image;
    if(DecodeInternal(data, size, image))
    {
        LOG_TRACE() << "Image was successfully decoded by NvJPEG decoder.";
        image.MoveToGpuMatAsync(decodedData, cudaStream_);
        return true;
    }
    else
    {
        LOG_ERROR() << "Failed to decode image with NvJPEG image decoder.";
        return false;
    }
}

bool NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG image decoder must be initialized before decoding.";
        return false;
    }
    if(DecodeInternal(data, size, decodedImage))
    {
        LOG_TRACE() << "Image was successfully decoded by NvJPEG decoder.";
        return true;
    }
    else
    {
        LOG_ERROR() << "Failed to decode image with NvJPEG image decoder.";
        return false;
    }
}

void NvJPEGImageDecoder::Initialize()
{
    LOG_TRACE() << "Initializing NvJPEG image decoder ...";
    if(!initialized_)
    {
        if(InitDecoder())
        {
            LOG_TRACE() << "NvJPEG image decoder was successfully initialized.";
        }
        else
        {
            LOG_ERROR() << "Failed to initialize NvJPEG image decoder.";
        }
    }
    else
    {
        LOG_WARNING() << "NvJPEG image decoder is already initialized.";
    }
}

bool NvJPEGImageDecoder::IsInitialized()
{
    return initialized_;
}

}
