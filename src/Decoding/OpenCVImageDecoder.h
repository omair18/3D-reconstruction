/**
 * @file OpenCVImageDecoder.h.
 *
 * @brief Declares a class of OpenCVImageDecoder. This decoder uses OpenCV backend and decodes most of image formats
 * on CPU.
 */

#ifndef OPENCV_IMAGE_DECODER_H
#define OPENCV_IMAGE_DECODER_H

#include "IImageDecoder.h"

/**
 * @namespace Decoding
 *
 * @brief Namespace of libdecoding library.
 */
namespace Decoding
{

/**
 * @class OpenCVImageDecoder
 *
 * @brief This decoder uses OpenCV backend and decodes most of image formats on CPU.
 */
class OpenCVImageDecoder final : public IImageDecoder
{
public:

    /**
     * @brief Default constructor.
     */
    OpenCVImageDecoder() = default;

    /**
     * @brief Default destructor.
     */
    ~OpenCVImageDecoder() noexcept(false) override;

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
     * @brief OpenCV image decoder doesn't require initialization. This method just creates records in a log file with
     * TRACE severity with messages that decoder was initialized.
     */
    void Initialize() override;

    /**
     * @brief Checks weather image decoder is initialized. OpenCV image decoder is always initialized.
     *
     * @return True is all cases.
     */
    bool IsInitialized() override;

};

}

#endif // OPENCV_IMAGE_DECODER_H