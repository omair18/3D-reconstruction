#include <cuda_runtime.h>

#include "NvJPEG2kImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

#include <opencv2/highgui.hpp>

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
bufferSize_(0),
initialized_(false),
buffer_(nullptr),
bufferPitch_(0)
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
        if (buffer_)
        {
            cudaStatus = cudaFree(buffer_);
            if(cudaStatus != cudaError_t::cudaSuccess)
            {
                LOG_ERROR() << "Failed to release NvJPEG2K buffer. Error " << static_cast<int>(cudaStatus) << ": "
                << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
                throw std::runtime_error("CUDA error");
            }
        }

        initialized_ = false;
    }
}

void NvJPEG2kImageDecoder::AllocateBuffer(int width, int height, int channels)
{
    if(!initialized_)
    {
        LOG_ERROR() << "NvJPEG2K image decoder must be initialized before allocating buffer.";
        return;
    }
    size_t imageSize = width * height * channels;
    cudaError_t cudaStatus;
    if(imageSize > bufferSize_)
    {
        LOG_TRACE() << "Allocating buffer for image " << width << "x" << height << " with " << channels << "channel(s)."
        << " Previous buffer size: " << bufferSize_;
        if (buffer_)
        {
            if((cudaStatus = cudaFree(buffer_)) != cudaError_t::cudaSuccess)
            {
                LOG_ERROR() << "Failed to release previous NvJPEG2K buffer. Error " << static_cast<int>(cudaStatus)
                << ": " << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
                throw std::runtime_error("CUDA error.");
            }
            else
            {
                LOG_TRACE() << "Previous NvJPEG2K buffer with size " << bufferSize_ << " bytes was successfully released.";
            }
        }
        if((cudaStatus = cudaMallocPitch(&buffer_, &bufferPitch_, width * channels, height)) != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to allocate pitched NvJPEG2K buffer. Error " << static_cast<int>(cudaStatus) << ": "
            << cudaGetErrorName(cudaStatus) << " - " << cudaGetErrorString(cudaStatus);
            throw std::runtime_error("CUDA error.");
        }
        else
        {
            bufferSize_ = bufferPitch_ * height;
            LOG_TRACE() << "Allocated net pitched NvJPEG2K buffer. Buffer size: " << bufferSize_ << ", pitch: "
            << bufferPitch_;
        }
    }
    else
    {
        LOG_TRACE() << "Using existing buffer for image decoding.";
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
                   static_cast<int>(imageInfo.num_components));

    nvjpeg2kImage_t decodedImage;

    decodedImage.num_components = 1;
    decodedImage.pixel_type = channelInfo.precision == 8 ? nvjpeg2kImageType_t::NVJPEG2K_UINT8 : nvjpeg2kImageType_t::NVJPEG2K_UINT16;
    decodedImage.pitch_in_bytes = &bufferPitch_;
    decodedImage.pixel_data = (void**)(&buffer_);

    CHECK_NVJPEG2K(nvjpeg2kDecode(handle_, decodeState_, jpeg2kStream_, &decodedImage, cudaStream_))

    DataStructures::CUDAImage CUDAImage;
    CUDAImage.gpuData_ = buffer_;
    CUDAImage.width_ = imageInfo.image_width;
    CUDAImage.height_ = imageInfo.image_height;
    CUDAImage.channels_ = imageInfo.num_components;
    CUDAImage.pitch_ = bufferPitch_;
    CUDAImage.elementType_ = DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U;
    CUDAImage.pitchedAllocation_ = true;
    CUDAImage.allocatedBytes_ = bufferPitch_ * imageInfo.image_height;

    outputImage.CopyFromCUDAImageAsync(CUDAImage, cudaStream_);

    cudaStreamSynchronize(cudaStream_);
    cv::Mat test;
    CUDAImage.CopyToCvMat(test);

    cv::imshow("after decoding", test);
    cv::waitKey(0);

    CUDAImage.gpuData_ = nullptr;

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
