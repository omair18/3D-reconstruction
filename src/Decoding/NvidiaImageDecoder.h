#pragma once
#include <nvjpeg.h>

#include "IImageDecoder.h"

class NvidiaImageDecoder final : public IImageDecoder
{
public:
    NvidiaImageDecoder();

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    bool IsInitialized() override;

    ~NvidiaImageDecoder() override;

private:
    void DecodeInternal(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& outputImage);

    void AllocateBuffer(int width, int height, int channels);

    void InitDecoder();

    nvjpegJpegState_t m_state;
    nvjpegHandle_t m_handle;
    nvjpegJpegState_t m_decoupledState;
    nvjpegBufferPinned_t m_pinnedBuffer;
    nvjpegBufferDevice_t m_deviceBuffer;
    nvjpegJpegStream_t  m_jpegStream;
    nvjpegDecodeParams_t m_decodeParams;
    nvjpegJpegDecoder_t m_decoder;
    cudaStream_t m_cudaStream;
    nvjpegImage_t m_imageBuffer;
    size_t m_bufferSize;
#if CUDART_VERSION >= 11000
    bool m_hardwareBackendAvailable;
#endif
    bool m_initialized;
};
