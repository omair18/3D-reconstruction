#ifndef NVJPEG_HARDWARE_IMAGE_DECODER_H
#define NVJPEG_HARDWARE_IMAGE_DECODER_H

#include "NvJPEGImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief
 */
namespace Decoding
{

/**
 * @class NvJPEGHardwareImageDecoder
 *
 * @brief
 */
class NvJPEGHardwareImageDecoder final : public NvJPEGImageDecoder
{
public:

    /**
     * @brief
     *
     * @param cudaStream
     */
    explicit NvJPEGHardwareImageDecoder(cudaStream_t cudaStream);

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
    ~NvJPEGHardwareImageDecoder() override = default;

private:

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
};

}

#endif // NVJPEG_HARDWARE_IMAGE_DECODER_H
