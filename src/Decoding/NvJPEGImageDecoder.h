/**
 * @file NvJPEGImageDecoder.h.
 *
 * @brief Declares a class of NvJPEGImageDecoder. This decoder uses NvJPEG backend and decodes image with JPEG format
 * on GPU.
 */

#ifndef NVJPEG_IMAGE_DECODER_H
#define NVJPEG_IMAGE_DECODER_H

#include <nvjpeg.h>

#include "IImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief Namespace of libdecoding library.
 */
namespace Decoding
{

/**
 * @class NvJPEGImageDecoder
 *
 * @brief This decoder uses NvJPEG backend and decodes image with JPEG format on GPU.
 */
class NvJPEGImageDecoder : public IImageDecoder
{
public:

    /**
     * @brief Constructor.
     *
     * @param cudaStream - CUDA stream of GPU processor
     */
    explicit NvJPEGImageDecoder(cudaStream_t cudaStream);

    /**
     * @brief Destructor.
     */
    ~NvJPEGImageDecoder() noexcept(false) override;

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

protected:

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedData - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    bool DecodeInternal(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& image);

    /**
     * @brief Allocates buffer for storing the result of decoding on GPU.
     *
     * @param width - Width of image that will be decoded
     * @param height - Height of image that will be decoded
     * @param channels - Number of channels of image that will be decoded
     */
    virtual void AllocateBuffer(int width, int height, int channels);

    /// Decoding state handle identifier. It is used to store intermediate information between decoding phases
    nvjpegJpegState_t state_{};

    /// State of NvJPEG decoder.
    nvjpegJpegState_t decoupledState_{};

    /// NvJPEG handle. For internal usage in NvJPEG library.
    nvjpegHandle_t handle_{};

    /// Host buffer for decoding image. For internal usage in NvJPEG library.
    nvjpegBufferPinned_t pinnedBuffer_{};

    /// Device buffer for decoding image. For internal usage in NvJPEG library.
    nvjpegBufferDevice_t deviceBuffer_{};

    /// Used to set decode-related tweaks.
    nvjpegDecodeParams_t decodeParams_{};

    /// NvJPEG decoder.
    nvjpegJpegDecoder_t decoder_{};

    /// NvJPEG stream for parsing image parameters on CPU.
    nvjpegJpegStream_t  jpegStream_{};

    /// CUDA stream of GPU processor
    cudaStream_t cudaStream_;

    /// Decoded image buffer
    nvjpegImage_t imageBuffer_{};

    /// Size of decoded image buffer in bytes
    size_t bufferSize_;

    /// Initialization flag
    bool initialized_;

private:

    /**
     * @brief Initializes internal structures of NvJPEG library and prepares decoder for image decoding.
     *
     * @return True if initialization was successful. Otherwise returns false.
     */
    bool InitDecoder();

};

}

#endif // NVJPEG_IMAGE_DECODER_H