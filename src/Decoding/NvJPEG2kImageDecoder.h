/**
 * @file NvJPEG2kImageDecoder.h.
 *
 * @brief @brief Declares a class of NvJPEG2kImageDecoder. This decoder uses NvJPEG2K backend and
 * decodes image with JPEG2000 format on GPU.
 */

#ifndef NVJPEG2K_DECODER_H
#define NVJPEG2K_DECODER_H

#include <nvjpeg2k.h>
#include <vector>

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
     * @brief Initializes backend of image decoder.
     */
    void Initialize() override;

    /**
     * @brief Checks weather image decoder is initialized.
     *
     * @return True if image decoder's backend is initialized. Otherwise returns false.
     */
    bool IsInitialized() override;

private:

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedData - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    bool DecodeInternal(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& outputImage);

    /**
     * @brief Allocates buffer for storing the result of decoding on GPU.
     *
     * @param width - Width of image that will be decoded
     * @param height - Height of image that will be decoded
     * @param channels - Number of channels of image that will be decoded
     * @param elementSize - Size of single element for storing
     */
    void AllocateBuffer(int width, int height, int channels, size_t elementSize);

    /**
     * @brief Initializes internal structures of NvJPEG library and prepares decoder for image decoding.
     *
     * @return True if initialization was successful. Otherwise returns false.
     */
    bool InitDecoder();

    /// CUDA stream of GPU processor
    cudaStream_t cudaStream_;

    /// NvJPEG2K handle. For internal usage.
    nvjpeg2kHandle_t handle_{};

    /// Contains intermediate decode information. For internal usage.
    nvjpeg2kDecodeState_t decodeState_{};

    /// NvJPEG2K stream for parsing image parameters on CPU.
    nvjpeg2kStream_t jpeg2kStream_{};

    /// Vector of GPU pointers to channel buffers
    std::vector<unsigned char*> bufferChannels_;

    /// Vector of channel buffers pitches
    std::vector<size_t> bufferChannelsPitches_;

    /// Vector of channel buffers sizes
    std::vector<size_t> bufferChannelsSizes_;

    /// Initialization flag
    bool initialized_;

};

}

#endif // NVJPEG2K_DECODER_H
