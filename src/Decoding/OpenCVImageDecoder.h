#ifndef OPENCV_IMAGE_DECODER_H
#define OPENCV_IMAGE_DECODER_H

#include "IImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief
 */
namespace Decoding
{

/**
 * @class OpenCVImageDecoder
 *
 * @brief
 */
class OpenCVImageDecoder final : public IImageDecoder
{
public:

    /**
     * @brief
     */
    OpenCVImageDecoder() = default;

    /**
     * @brief
     */
    ~OpenCVImageDecoder() override = default;

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
     * @param width
     * @param height
     * @param channels
     */
    void AllocateBuffer(int width, int height, int channels) override;
};

}

#endif // OPENCV_IMAGE_DECODER_H