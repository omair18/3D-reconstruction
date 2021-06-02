#include <cuda_runtime.h>

#include "NvJPEG2kImageDecoder.h"
#include "NvJPEG2KChannelMerging.h"
#include "CUDAImage.h"
#include "Logger.h"

#define CHECK_NVJPEG2K(status)                                              \
{                                                                           \
    if (status != nvjpeg2kStatus_t::NVJPEG2K_STATUS_SUCCESS)                \
    {                                                                       \
        LOG_ERROR() << "NvJPEG2K error " << static_cast<int>(status);       \
        return false;                                                       \
    }                                                                       \
}

namespace Decoding
{

NvJPEG2kImageDecoder::NvJPEG2kImageDecoder(cudaStream_t cudaStream) :
cudaStream_(cudaStream),
initialized_(false),
bufferChannels_(0)
{

}

NvJPEG2kImageDecoder::~NvJPEG2kImageDecoder() noexcept(false)
{
    if(initialized_)
    {
        nvjpeg2kStatus_t status;

        if((status = nvjpeg2kStreamDestroy(jpeg2kStream_)) != nvjpeg2kStatus_t::NVJPEG2K_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG2K decoding stream. Error: " << static_cast<int>(status);
        }

        if((status = nvjpeg2kDecodeStateDestroy(decodeState_)) != nvjpeg2kStatus_t::NVJPEG2K_STATUS_SUCCESS)
        {
            LOG_ERROR() << "Failed to destroy NvJPEG2K decoding decoder state structure. Error: " << static_cast<int>(status);
        }

        cudaError_t cudaStatus;
        for(auto& channelBuffer : bufferChannels_)
        {
            if (channelBuffer)
            {
                cudaStatus = cudaFree(channelBuffer);
                if(cudaStatus != cudaError_t::cudaSuccess)
                {
                    LOG_ERROR() << "Failed to release NvJPEG2K buffer. Error " << static_cast<int>(cudaStatus) << ": "
                                << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA error");
                }
            }
        }

        initialized_ = false;
    }
}

void NvJPEG2kImageDecoder::AllocateBuffer(int width, int height, int channels, size_t elementSize)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG2K image decoder must be initialized before allocating buffer.";
        return;
    }
    size_t imageChannelSize = width * height;
    cudaError_t cudaStatus;

    for(int i = 0; i < channels; ++i)
    {
        if(i == bufferChannels_.size())
        {
            LOG_TRACE() << "Allocating buffer fow channel " << i << "of image " << width << "x" << height;
            unsigned char* gpuData = nullptr;
            size_t gpuDataSize = 0;
            size_t gpuDataPitch = 0;

            cudaStatus = cudaMallocPitch(&gpuData, &gpuDataPitch, width * elementSize, height);
            if(cudaStatus == cudaError_t::cudaSuccess)
            {
                LOG_TRACE() << "Successfully allocated NvJPEG2K channel buffer for image " << width << "x" << height;
                gpuDataSize = gpuDataPitch * height;
                bufferChannels_.push_back(gpuData);
                bufferChannelsSizes_.push_back(gpuDataSize);
                bufferChannelsPitches_.push_back(gpuDataPitch);
            }
            else
            {
                LOG_ERROR() << "Failed to allocate NvJPEG2K channel buffer for image " << width << "x" << height << "." \
                " CUDA error " << static_cast<int>(cudaStatus) << ": " << cudaGetErrorName(cudaStatus) << " - "
                << cudaGetErrorString(cudaStatus);
                throw std::runtime_error("CUDA error.");
            }
        }
        else
        {
            if(bufferChannelsSizes_[i] < imageChannelSize)
            {
                LOG_TRACE() << "Channel buffer " << i << " is too small for image " << width << "x" <<height << ". " \
                "Reallocating buffer for this channel";

                cudaStatus = cudaFree(bufferChannels_[i]);
                if(cudaStatus != cudaError_t::cudaSuccess)
                {
                    LOG_ERROR() << "Failed to release previous channel buffer. CUDA error " << static_cast<int>(cudaStatus)
                    << ": " << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA error");
                }

                cudaStatus = cudaMallocPitch(&bufferChannels_[i], &bufferChannelsPitches_[i], width * elementSize, height);
                if(cudaStatus == cudaError_t::cudaSuccess)
                {
                    LOG_TRACE() << "Successfully allocated NvJPEG2K channel buffer for image " << width << "x" << height;
                    bufferChannelsSizes_[i] = bufferChannelsPitches_[i] * height;
                }
                else
                {
                    LOG_ERROR() << "Failed to allocate NvJPEG2K channel buffer for image " << width << "x" << height
                    << ". CUDA error " << static_cast<int>(cudaStatus) << ": " << cudaGetErrorName(cudaStatus) << " - "
                    << cudaGetErrorString(cudaStatus);
                    throw std::runtime_error("CUDA error.");
                }
            }
            else
            {
                LOG_TRACE() << "Using existing buffer for image decoding.";
            }
        }
    }
}

bool NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG2K image decoder must be initialized before decoding.";
        return false;
    }
    DataStructures::CUDAImage image;
    if(DecodeInternal(data, size, image))
    {
        LOG_TRACE() << "Image was successfully decoded with NvJPEG2K decoder.";
        image.MoveToCvMatAsync(decodedData, cudaStream_);
        return true;
    }
    else
    {
        LOG_ERROR() << "Failed to decode image with NvJPEG2K decoder.";
        return false;
    }
}

bool NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG2K image decoder must be initialized before decoding.";
        return false;
    }
    DataStructures::CUDAImage image;
    if(DecodeInternal(data, size, image))
    {
        LOG_TRACE() << "Image was successfully decoded by NvJPEG2K decoder.";
        image.MoveToGpuMatAsync(decodedData, cudaStream_);
        return true;
    }
    else
    {
        LOG_ERROR() << "Failed to decode image with NvJPEG2K image decoder.";
        return false;
    }
}

bool NvJPEG2kImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG2K image decoder must be initialized before decoding.";
        return false;
    }
    if(DecodeInternal(data, size, decodedImage))
    {
        LOG_TRACE() << "Image was successfully decoded by NvJPEG2K decoder.";
        return true;
    }
    else
    {
        LOG_ERROR() << "Failed to decode image with NvJPEG2K image decoder.";
        return false;
    }
}

void NvJPEG2kImageDecoder::Initialize()
{
    LOG_TRACE() << "Initializing NvJPEG2K image decoder ...";
    if(!initialized_)
    {
        if(InitDecoder())
        {
            LOG_TRACE() << "NvJPEG2K image decoder was successfully initialized.";
        }
        else
        {
            LOG_ERROR() << "Failed to initialize NvJPEG2K image decoder.";
        }
    }
    else
    {
        LOG_WARNING() << "NvJPEG2K image decoder is already initialized.";
    }
}

bool NvJPEG2kImageDecoder::IsInitialized()
{
    return initialized_;
}

bool NvJPEG2kImageDecoder::DecodeInternal(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage& outputImage)
{
    nvjpeg2kImageInfo_t imageInfo;
    nvjpeg2kImageComponentInfo_t channelInfo;

    CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle_, data, size, 0, 0, jpeg2kStream_))

    CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(jpeg2kStream_, &imageInfo))

    CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(jpeg2kStream_, &channelInfo, 0))

    LOG_TRACE() << "Decoding image " << imageInfo.image_width << "x" << imageInfo.image_height << " with " << imageInfo.num_components << " channel(s) ...";

    AllocateBuffer(static_cast<int>(imageInfo.image_width),
                   static_cast<int>(imageInfo.image_height),
                   static_cast<int>(imageInfo.num_components),
                   channelInfo.precision / 8);

    nvjpeg2kImage_t decodedImage;

    decodedImage.num_components = imageInfo.num_components;
    decodedImage.pixel_type = channelInfo.precision == 8 ? nvjpeg2kImageType_t::NVJPEG2K_UINT8 : nvjpeg2kImageType_t::NVJPEG2K_UINT16;
    decodedImage.pitch_in_bytes = bufferChannelsPitches_.data();
    decodedImage.pixel_data = (void**)bufferChannels_.data();

    CHECK_NVJPEG2K(nvjpeg2kDecode(handle_, decodeState_, jpeg2kStream_, &decodedImage, cudaStream_))

    outputImage.AllocateAsync(imageInfo.image_width, imageInfo.image_height, imageInfo.num_components,
                              channelInfo.precision == 8 ? DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U :
                              DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_16U, true, cudaStream_);

    MergeChannels(imageInfo.image_width, imageInfo.image_height, imageInfo.num_components,
                  channelInfo.precision, outputImage.pitch_, (void**)bufferChannels_.data(), bufferChannelsPitches_.data(),
                  outputImage.gpuData_, cudaStream_);

    return true;
}

bool NvJPEG2kImageDecoder::InitDecoder()
{
    CHECK_NVJPEG2K(nvjpeg2kCreate(nvjpeg2kBackend_t::NVJPEG2K_BACKEND_DEFAULT, nullptr, nullptr, &handle_))
    CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &decodeState_))
    CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&jpeg2kStream_))

    initialized_ = true;

    return true;
}

}
