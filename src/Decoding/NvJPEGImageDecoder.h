#ifndef NVJPEG_IMAGE_DECODER_H
#define NVJPEG_IMAGE_DECODER_H

#include <nvjpeg.h>

#include "IImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief
 */
namespace Decoding
{

/**
 * @class NvJPEGImageDecoder
 *
 * @brief
 */
class NvJPEGImageDecoder : public IImageDecoder
{
public:

    /**
     * @brief
     *
     * @param cudaStream
     */
    explicit NvJPEGImageDecoder(cudaStream_t cudaStream);

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param decodedData
     */
    void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param decodedData
     */
    void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param decodedImage
     */
    void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) override;

    /**
     * @brief
     */
    void Initialize() override;

    /**
     * @brief
     *
     * @return
     */
    bool IsInitialized() override;

    /**
     * @brief
     */
    ~NvJPEGImageDecoder() override;

protected:

    ///
    nvjpegJpegState_t state_{};

    ///
    nvjpegJpegState_t decoupledState_{};

    ///
    nvjpegHandle_t handle_{};

    ///
    nvjpegBufferPinned_t pinnedBuffer_{};

    ///
    nvjpegBufferDevice_t deviceBuffer_{};

    ///
    nvjpegDecodeParams_t decodeParams_{};

    ///
    nvjpegJpegDecoder_t decoder_{};

    ///
    nvjpegJpegStream_t  jpegStream_{};

    ///
    cudaStream_t cudaStream_;

    ///
    nvjpegImage_t imageBuffer_{};

    ///
    size_t bufferSize_;

    ///
    bool initialized_;

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param image
     */
    void DecodeInternal(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& image);

    /**
     * @brief
     *
     * @param width
     * @param height
     * @param channels
     */
    void AllocateBuffer(int width, int height, int channels) override;

private:

    /**
     * @brief
     */
    void InitDecoder();

};

}

#endif // NVJPEG_IMAGE_DECODER_H