#include <opencv2/core/cuda.hpp>
#include <nppi_geometry_transforms.h>

#include "NvJPEGImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

#define CHECK_NVJPEG(status)                                                \
{                                                                           \
    if (status != NVJPEG_STATUS_SUCCESS)                                    \
    {                                                                       \
        LOG_ERROR() << "NVJPEG error " << static_cast<int>(status);         \
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

NvJPEGImageDecoder::NvJPEGImageDecoder(cudaStream_t& cudaStream) :
initialized_(false),
cudaStream_(cudaStream)
{
    bufferSize_ = 0;
    resizeBufferSize_ = 0;
    nppStreamContext_.hStream = cudaStream;
    std::memset(&imageBuffer_, 0, sizeof(imageBuffer_));
    std::memset(&resizeBuffer_, 0, sizeof(resizeBuffer_));
}

NvJPEGImageDecoder::~NvJPEGImageDecoder()
{
    nvjpegDecodeParamsDestroy(decodeParams_);
    nvjpegJpegStreamDestroy(jpegStream_);
    nvjpegBufferPinnedDestroy(pinnedBuffer_);
    nvjpegBufferDeviceDestroy(deviceBuffer_);
    nvjpegJpegStateDestroy(decoupledState_);
    nvjpegDecoderDestroy(decoder_);

    nvjpegJpegStateDestroy(state_);
    nvjpegDestroy(handle_);

    for (auto &channel : imageBuffer_.channel)
    {
        if (channel)
        {
            cudaFree(channel);
        }
    }

    for (auto &channel : resizeBuffer_.channel)
    {
        if (channel)
        {
            cudaFree(channel);
        }
    }
}

void NvJPEGImageDecoder::InitDecoder()
{
    nvjpegStatus_t status = nvjpegCreateEx(nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, nullptr,
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

    CHECK_NVJPEG(nvjpegDecoderCreate(handle_, nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, &decoder_));
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

void NvJPEGImageDecoder::AllocateBuffer(int width, int height, int channels)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    size_t imageSize = width * height * channels;
    if(imageSize > bufferSize_)
    {
        if (imageBuffer_.channel[0])
        {
            CHECK_CUDA(cudaFree(imageBuffer_.channel[0]));
        }
        cudaMallocPitch(&imageBuffer_.channel[0], &imageBuffer_.pitch[0], width, height);
        bufferSize_ = imageSize;
    }
}

void NvJPEGImageDecoder::AllocateResizeBuffer(int width, int height, int channels)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    size_t imageSize = width * height * channels;
    if(imageSize > resizeBufferSize_)
    {
        if (resizeBuffer_.channel[0])
        {
            CHECK_CUDA(cudaFree(resizeBuffer_.channel[0]));
        }
        cudaMallocPitch(&resizeBuffer_.channel[0], &resizeBuffer_.pitch[0], width, height);
        resizeBufferSize_ = imageSize;
    }
}

void NvJPEGImageDecoder::DecodeInternal(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage& image)
{
    unsigned int channels;
    unsigned int width;
    unsigned int height;

    CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, reinterpret_cast<const unsigned char *>(data), size, 0, 0, jpegStream_));

    nvjpegJpegStreamGetComponentsNum(jpegStream_, &channels);

    nvjpegJpegStreamGetComponentDimensions(jpegStream_, 0, &width, &height);

    AllocateBuffer(width, height, channels);

    CHECK_NVJPEG(nvjpegDecodeJpegHost(handle_, decoder_, decoupledState_, decodeParams_, jpegStream_));

    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, decoder_, decoupledState_, jpegStream_, cudaStream_));

    CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, decoder_, decoupledState_, &imageBuffer_, cudaStream_));

    DataStructures::CUDAImage decodedImage;
    decodedImage.gpuData_ = imageBuffer_.channel[0];
    decodedImage.width_ = width;
    decodedImage.height_ = height;
    decodedImage.channels_ = channels;
    decodedImage.pitch_ = imageBuffer_.pitch[0];
    decodedImage.elementSize_ = sizeof(unsigned char);
    decodedImage.pitchedAllocation_ = true;

    image.CopyFromCUDAImageAsync(decodedImage, cudaStream_);

    CHECK_CUDA(cudaStreamSynchronize(cudaStream_));

    decodedImage.gpuData_ = nullptr;
}

void NvJPEGImageDecoder::DecodeInternalWithResize(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &image, size_t outputWidth, size_t outputHeight)
{
    unsigned int channels;
    unsigned int width;
    unsigned int height;

    CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, reinterpret_cast<const unsigned char *>(data), size, 0, 0, jpegStream_));

    nvjpegJpegStreamGetComponentsNum(jpegStream_, &channels);

    nvjpegJpegStreamGetComponentDimensions(jpegStream_, 0, &width, &height);

    NppStatus resizeStatus = NppStatus::NPP_NO_ERROR;

    NppiSize inputSize { .width = static_cast<int>(width), .height = static_cast<int>(height)};
    NppiRect inputRoi { .x = 0, .y = 0, .width = inputSize.width, .height = inputSize.height};

    NppiSize outputSize { .width = static_cast<int>(outputWidth), .height = static_cast<int>(outputHeight)};
    NppiRect outputRoi { .x = 0, .y = 0, .width = outputSize.width, .height = outputSize.height};

    AllocateBuffer(width, height, channels);

    AllocateResizeBuffer(outputWidth, outputHeight, channels);

    CHECK_NVJPEG(nvjpegDecodeJpegHost(handle_, decoder_, decoupledState_, decodeParams_, jpegStream_));

    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, decoder_, decoupledState_, jpegStream_, cudaStream_));

    CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, decoder_, decoupledState_, &imageBuffer_, cudaStream_));

    switch (channels)
    {
        case 1:
        {
            resizeStatus = nppiResize_8u_C1R_Ctx(imageBuffer_.channel[0], imageBuffer_.pitch[0], inputSize, inputRoi,
                                  resizeBuffer_.channel[0], resizeBuffer_.pitch[0], outputSize, outputRoi, NPPI_INTER_LINEAR, nppStreamContext_);
        } break;

        case 3:
        {
            resizeStatus = nppiResize_8u_C3R_Ctx(imageBuffer_.channel[0], imageBuffer_.pitch[0], inputSize, inputRoi,
                                  resizeBuffer_.channel[0], resizeBuffer_.pitch[0], outputSize, outputRoi, NPPI_INTER_LINEAR, nppStreamContext_);
        } break;

        case 4:
        {
            resizeStatus = nppiResize_8u_C4R_Ctx(imageBuffer_.channel[0], imageBuffer_.pitch[0], inputSize, inputRoi,
                                  resizeBuffer_.channel[0], resizeBuffer_.pitch[0], outputSize, outputRoi, NPPI_INTER_LINEAR, nppStreamContext_);
        } break;

        default:
        {
            LOG_ERROR() << "Unsupported channels amount: " << channels;
            resizeStatus = NppStatus::NPP_ERROR;
        }
    }
    if(resizeStatus != NppStatus::NPP_NO_ERROR)
    {
        LOG_ERROR() << "";
        return;
    }

    DataStructures::CUDAImage decodedImage;
    decodedImage.gpuData_ = resizeBuffer_.channel[0];
    decodedImage.width_ = outputWidth;
    decodedImage.height_ = outputHeight;
    decodedImage.channels_ = channels;
    decodedImage.pitch_ = resizeBuffer_.pitch[0];
    decodedImage.elementSize_ = sizeof(unsigned char);
    decodedImage.pitchedAllocation_ = true;

    image.CopyFromCUDAImageAsync(decodedImage, cudaStream_);

    CHECK_CUDA(cudaStreamSynchronize(cudaStream_));

    decodedImage.gpuData_ = nullptr;
}

void NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    DataStructures::CUDAImage image;
    DecodeInternal(data, size, image);
    cv::cuda::GpuMat gpuMat(image.height_, image.width_, CV_8UC3, image.gpuData_, image.pitch_);
    gpuMat.download(decodedData);
    image.gpuData_ = nullptr;
}

void NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    DataStructures::CUDAImage image;
    DecodeInternal(data, size, image);
    decodedData = cv::cuda::GpuMat(image.height_, image.width_, CV_8UC3, image.gpuData_, image.pitch_);
    image.gpuData_ = nullptr;
}

void NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedImage, size_t outputWidth, size_t outputHeight)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    DataStructures::CUDAImage image;
    DecodeInternalWithResize(data, size, image, outputWidth, outputHeight);
    cv::cuda::GpuMat gpuMat(image.height_, image.width_, CV_8UC3, image.gpuData_, image.pitch_);
    gpuMat.download(decodedImage);
    image.gpuData_ = nullptr;
}

void NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedImage, size_t outputWidth, size_t outputHeight)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    DataStructures::CUDAImage image;
    DecodeInternalWithResize(data, size, image, outputWidth, outputHeight);
    decodedImage = cv::cuda::GpuMat(image.height_, image.width_, CV_8UC3, image.gpuData_, image.pitch_);
    image.gpuData_ = nullptr;
}

void NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    DecodeInternal(data, size, decodedImage);
}

void NvJPEGImageDecoder::Decode(const unsigned char *data, unsigned long long int size, DataStructures::CUDAImage &decodedImage, size_t outputWidth, size_t outputHeight)
{
    if(!initialized_)
    {
        LOG_ERROR() << "";
        return;
    }
    DecodeInternalWithResize(data, size, decodedImage, outputWidth, outputHeight);
}

void NvJPEGImageDecoder::Initialize()
{
    InitDecoder();
}

bool NvJPEGImageDecoder::IsInitialized()
{
    return initialized_;
}

}
