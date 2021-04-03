/**
 * @file NvJPEGHardwareImageDecoder.h.
 *
 * @brief Declares a class of NvJPEGHardwareImageDecoder. This decoder uses NvJPEG hardware backend and decodes image with JPEG format
 * on GPU.
 */

#ifndef NVJPEG_HARDWARE_IMAGE_DECODER_H
#define NVJPEG_HARDWARE_IMAGE_DECODER_H

#include "NvJPEGImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief Namespace of libdecoding library.
 */
namespace Decoding
{

/**
 * @class NvJPEGHardwareImageDecoder
 *
 * @brief This decoder uses NvJPEG hardware backend and decodes image with JPEG format on GPU.
 */
class NvJPEGHardwareImageDecoder final : public NvJPEGImageDecoder
{
public:

    /**
     * @brief Constructor.
     *
     * @param cudaStream - CUDA stream of GPU processor
     */
    explicit NvJPEGHardwareImageDecoder(cudaStream_t cudaStream);

    /**
     * @brief Destructor.
     */
    ~NvJPEGHardwareImageDecoder() noexcept(false) override;

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
     * @brief Allocates buffer for storing the result of decoding on GPU.
     *
     * @param width - Width of image that will be decoded
     * @param height - Height of image that will be decoded
     * @param channels - Number of channels of image that will be decoded
     */
    void AllocateBuffer(int width, int height, int channels) override;

    /**
     * @brief Initializes internal structures of NvJPEG library and prepares decoder for image decoding.
     *
     * @return True if initialization was successful. Otherwise returns false.
     */
    bool InitDecoder();
};

}

#endif // NVJPEG_HARDWARE_IMAGE_DECODER_H
