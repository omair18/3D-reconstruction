/**
 * @file IImageDecoder.h.
 *
 * @brief Declares a class of IImageDecoder. This is a base class for all image decoders.
 */

#ifndef INTERFACE_IMAGE_DECODER_H
#define INTERFACE_IMAGE_DECODER_H

// forward declaration for cv::Mat and cv::cuda::GpuMat
namespace cv
{
    class Mat;
    namespace cuda
    {
        class GpuMat;
    }
}

// forward declaration for DataStructures::CUDAImage
namespace DataStructures
{
    struct CUDAImage;
}

/**
 * @namespace Decoding
 *
 * @brief Namespace of libdecoding library.
 */
namespace Decoding
{

/**
 * @class IImageDecoder
 *
 * @brief This is a base class for all image decoders.
 */
class IImageDecoder
{
public:

    /**
     * @brief Default constructor.
     */
    IImageDecoder() = default;

    /**
     * @brief Default destructor.
     */
    virtual ~IImageDecoder() noexcept(false) {;};

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedImage - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    virtual bool Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage) = 0;

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedImage - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    virtual bool Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage) = 0;

    /**
     * @brief Decodes image from raw host pointer and stores decoded image to decodedImage-param.
     *
     * @param data - Host raw pointer to data for decoding
     * @param size - Size of data in bytes
     * @param decodedImage - Decoded image
     * @return True if decoding was successful. Otherwise returns false.
     */
    virtual bool Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) = 0;

    /**
     * @brief Initializes backend of image decoder.
     */
    virtual void Initialize() = 0;

    /**
     * @brief Checks weather image decoder is initialized.
     *
     * @return True if image decoder's backend is initialized. Otherwise returns false.
     */
    virtual bool IsInitialized() = 0;

};

}

#endif // INTERFACE_IMAGE_DECODER_H