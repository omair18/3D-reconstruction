#ifndef NVJPEG2K_DECODER_H
#define NVJPEG2K_DECODER_H

#include <nvjpeg2k.h>

#include "IImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief
 */
namespace Decoding
{

/**
 * @class NvJPEG2kImageDecoder
 *
 * @brief
 */
class NvJPEG2kImageDecoder final : public IImageDecoder
{
public:

    /**
     * @brief
     *
     * @param cudaStream
     */
    explicit NvJPEG2kImageDecoder(cudaStream_t cudaStream);

    /**
     * @brief
     */
    ~NvJPEG2kImageDecoder() override;

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

private:

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param outputImage
     */
    void DecodeInternal(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& outputImage);

    /**
     * @brief
     *
     * @param width
     * @param height
     * @param channels
     */
    void AllocateBuffer(int width, int height, int channels) override;

    /**
     * @brief
     */
    void InitDecoder();

    ///
    cudaStream_t cudaStream_;

    ///
    nvjpeg2kStatus_t state_{};

    ///
    nvjpeg2kStatus_t decoupledState_{};

    ///
    nvjpeg2kHandle_t handle_{};

    ///
    nvjpeg2kDecodeState_t decodeState_{};

    ///
    nvjpeg2kStream_t jpegStream_;

    ///
    nvjpeg2kImage_t imageBuffer_{};

    ///
    size_t bufferSize_;

    bool initialized_;

};

}

#endif // NVJPEG2K_DECODER_H
