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
    explicit NvJPEGImageDecoder(cudaStream_t& cudaStream);

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) override;

    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage, size_t outputWidth, size_t outputHeight) override;

    void Initialize() override;

    bool IsInitialized() override;

    ~NvJPEGImageDecoder() override;

protected:

    nvjpegJpegState_t state_{};
    nvjpegJpegState_t decoupledState_{};
    nvjpegHandle_t handle_{};

    nvjpegBufferPinned_t pinnedBuffer_{};
    nvjpegBufferDevice_t deviceBuffer_{};

    nvjpegDecodeParams_t decodeParams_{};
    nvjpegJpegDecoder_t decoder_{};

    nvjpegJpegStream_t  jpegStream_{};
    cudaStream_t cudaStream_;

    nvjpegImage_t imageBuffer_{};
    size_t bufferSize_;

    NppStreamContext nppStreamContext_;
    nvjpegImage_t resizeBuffer_{};
    size_t resizeBufferSize_;

    bool initialized_;

    void DecodeInternal(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& image);

    void DecodeInternalWithResize(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& image, size_t outputWidth, size_t outputHeight);

    void AllocateBuffer(int width, int height, int channels) override;

    void AllocateResizeBuffer(int width, int height, int channels);

private:

    void InitDecoder();

};

}

#endif // NVJPEG_IMAGE_DECODER_H