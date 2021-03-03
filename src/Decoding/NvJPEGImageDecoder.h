#ifndef NVJPEG_IMAGE_DECODER_H
#define NVJPEG_IMAGE_DECODER_H

#include <nvjpeg.h>
#include <npp.h>

#include "IImageDecoder.h"

namespace Decoding
{

class NvJPEGImageDecoder : public IImageDecoder
{
public:
    NvJPEGImageDecoder();

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Initialize() override;

    bool IsInitialized() override;

    ~NvJPEGImageDecoder() override;

private:
    void DecodeInternal(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& outputImage);

    void AllocateBuffer(int width, int height, int channels);

    void InitDecoder();

    nvjpegJpegState_t m_state;
    nvjpegJpegState_t m_decoupledState;
    nvjpegHandle_t m_handle;

    nvjpegBufferPinned_t m_pinnedBuffer;
    nvjpegBufferDevice_t m_deviceBuffer;

    nvjpegDecodeParams_t m_decodeParams;
    nvjpegJpegDecoder_t m_decoder;

    nvjpegJpegStream_t  m_jpegStream;
    cudaStream_t m_cudaStream;

    nvjpegImage_t m_imageBuffer;
    size_t m_bufferSize;

    NppStatus resizeStatus_;
    NppiSize outputSize_;

    bool m_hardwareBackendAvailable;
    bool m_initialized;
};

}

#endif // NVJPEG_IMAGE_DECODER_H