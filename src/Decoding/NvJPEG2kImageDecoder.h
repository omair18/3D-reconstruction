/**
 * @file NvJPEG2kImageDecoder.h.
 *
 * @brief @brief Declares a class of NvJPEG2kImageDecoder. This decoder uses NvJPEG2K backend and
 * decodes image with JPEG2000 format on GPU.
 */

#ifndef NVJPEG2K_DECODER_H
#define NVJPEG2K_DECODER_H

#include <nvjpeg2k.h>

#include "IImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief Namespace of libdecoding library.
 */
namespace Decoding
{

/**
 * @class NvJPEG2kImageDecoder
 *
 * @brief This decoder uses NvJPEG2K backend and decodes image with JPEG2000 format on GPU.
 */
class NvJPEG2kImageDecoder final : public IImageDecoder
{
public:

    /**
     * @brief Constructor.
     *
     * @param cudaStream - CUDA stream of GPU processor
     */
    explicit NvJPEG2kImageDecoder(cudaStream_t cudaStream);

    /**
     * @brief Destructor.
     */
    ~NvJPEG2kImageDecoder() noexcept(false) override;

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedData - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    bool Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedData) override;

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedData - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    bool Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedData) override;

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedData - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    bool Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) override;

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
     * @return
     */
    bool DecodeInternal(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& outputImage);

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
     *
     * @return
     */
    bool InitDecoder();

    ///
    cudaStream_t cudaStream_;

    ///
    nvjpeg2kHandle_t handle_{};

    ///
    nvjpeg2kDecodeState_t decodeState_{};

    ///
    nvjpeg2kStream_t jpeg2kStream_{};

    ///
    unsigned char* buffer_;

    ///
    size_t bufferSize_;

    ///
    size_t bufferPitch_;

    ///
    bool initialized_;

};

}

#endif // NVJPEG2K_DECODER_H
