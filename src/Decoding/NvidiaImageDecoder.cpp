#include <opencv2/core/cuda.hpp>

#include "NvidiaImageDecoder.h"
#include "Logger.h"

#define CHECK_NVJPEG(status)                                                    \
    {                                                                           \
        if (status != NVJPEG_STATUS_SUCCESS)                                    \
        {                                                                       \
             LOG_ERROR("NVJPEG error %d", (int)(status));                       \
             return;                                                            \
        }                                                                       \
    }

#define CHECK_CUDA(status)                                                      \
    {                                                                           \
        if (status != cudaSuccess)                                              \
        {                                                                       \
            LOG_ERROR("CUDA Runtime error %d", (int)(status));                  \
            return;                                                             \
        }                                                                       \
    }

NvidiaImageDecoder::NvidiaImageDecoder() :
m_initialized(false)
{
    m_bufferSize = 0;
    std::memset(&m_imageBuffer, 0, sizeof(m_imageBuffer));
    InitDecoder();
}

NvidiaImageDecoder::~NvidiaImageDecoder()
{
    nvjpegDecodeParamsDestroy(m_decodeParams);
    nvjpegJpegStreamDestroy(m_jpegStream);
    nvjpegBufferPinnedDestroy(m_pinnedBuffer);
    nvjpegBufferDeviceDestroy(m_deviceBuffer);
    nvjpegJpegStateDestroy(m_decoupledState);
    nvjpegDecoderDestroy(m_decoder);

    nvjpegJpegStateDestroy(m_state);
    nvjpegDestroy(m_handle);
    cudaStreamDestroy(m_cudaStream);

    for (auto & channel : m_imageBuffer.channel)
    {
        if (channel)
        {
            cudaFree(channel);
        }
    }
}

void NvidiaImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::Mat &decodedData)
{
    cv::cuda::GpuMat decodedImage;
    DecodeInternal(data, size, decodedImage);
    decodedImage.download(decodedData);
}

void NvidiaImageDecoder::Decode(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &decodedData)
{
    cv::cuda::GpuMat decodedImage;
    DecodeInternal(data, size, decodedImage);
    decodedData = decodedImage.clone();
}

void NvidiaImageDecoder::InitDecoder()
{
    cudaStreamCreateWithFlags(&m_cudaStream, cudaStreamNonBlocking);
#if CUDART_VERSION >= 11000
    nvjpegStatus_t status = nvjpegCreateEx(nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE, nullptr,
                                           nullptr, NVJPEG_FLAGS_DEFAULT, &m_handle);
    m_hardwareBackendAvailable = true;
    if(status == nvjpegStatus_t::NVJPEG_STATUS_ARCH_MISMATCH)
    {
        m_hardwareBackendAvailable = false;
        status = nvjpegCreateEx(nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, nullptr,
                                nullptr, NVJPEG_FLAGS_DEFAULT, &m_handle);
        LOG_WARNING("Failed to created nvidia decoder with hardware backend. Creating with default backend");
    }
#else
    nvjpegStatus_t status = nvjpegCreateEx(nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, nullptr,
                   nullptr, NVJPEG_FLAGS_DEFAULT, &m_handle);
#endif
    if (status != nvjpegStatus_t::NVJPEG_STATUS_SUCCESS)
    {
        LOG_ERROR("Failed to create nvidia decoder.");
    }
    else
    {
        CHECK_NVJPEG(nvjpegJpegStateCreate(m_handle, &m_state));
    }
#if CUDART_VERSION >= 11000
    if(m_hardwareBackendAvailable)
    {
        CHECK_NVJPEG(nvjpegDecoderCreate(m_handle, nvjpegBackend_t::NVJPEG_BACKEND_HARDWARE, &m_decoder));
    }
    else
    {
        CHECK_NVJPEG(nvjpegDecoderCreate(m_handle, nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, &m_decoder));
    }
#else
    CHECK_NVJPEG(nvjpegDecoderCreate(m_handle, nvjpegBackend_t::NVJPEG_BACKEND_HYBRID, &m_decoder));
#endif
    CHECK_NVJPEG(nvjpegDecoderStateCreate(m_handle, m_decoder, &m_decoupledState));
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(m_handle, nullptr, &m_pinnedBuffer));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(m_handle, nullptr, &m_deviceBuffer));
    CHECK_NVJPEG(nvjpegJpegStreamCreate(m_handle, &m_jpegStream));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(m_handle, &m_decodeParams));

    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(m_decoupledState, m_deviceBuffer));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(m_decoupledState, m_pinnedBuffer));
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(m_decodeParams, nvjpegOutputFormat_t::NVJPEG_OUTPUT_BGRI));
    m_initialized = true;
}

void NvidiaImageDecoder::AllocateBuffer(int width, int height, int channels)
{
    size_t imageSize = width * height * channels;
    if(imageSize > m_bufferSize)
    {
        if (m_imageBuffer.channel[0])
        {
            CHECK_CUDA(cudaFree(m_imageBuffer.channel[0]));
        }
        cudaMalloc(reinterpret_cast<void **>(&m_imageBuffer.channel[0]), imageSize);
        m_imageBuffer.pitch[0] = width * channels;
        m_bufferSize = imageSize;
    }
}

void NvidiaImageDecoder::DecodeInternal(const unsigned char *data, unsigned long long int size, cv::cuda::GpuMat &outputImage)
{
    unsigned int channels;
    unsigned int width;
    unsigned int height;

    CHECK_NVJPEG(nvjpegJpegStreamParse(m_handle, reinterpret_cast<const unsigned char *>(data), size, 0, 0, m_jpegStream));

    nvjpegJpegStreamGetComponentsNum(m_jpegStream, &channels);

    nvjpegJpegStreamGetComponentDimensions(m_jpegStream, 0, &width, &height);

    AllocateBuffer(width, height, channels);

    CHECK_NVJPEG(nvjpegDecodeJpegHost(m_handle, m_decoder, m_decoupledState, m_decodeParams, m_jpegStream));

    CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(m_handle, m_decoder, m_decoupledState, m_jpegStream, m_cudaStream));

    CHECK_NVJPEG(nvjpegDecodeJpegDevice(m_handle, m_decoder, m_decoupledState, &m_imageBuffer, m_cudaStream));

    CHECK_CUDA(cudaStreamSynchronize(m_cudaStream));

    outputImage = cv::cuda::GpuMat(height, width, CV_8UC3, m_imageBuffer.channel[0]);
}

bool NvidiaImageDecoder::IsInitialized()
{
    return m_initialized;
}


