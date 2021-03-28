#ifndef INTERFACE_IMAGE_DECODER_H
#define INTERFACE_IMAGE_DECODER_H

//
namespace cv
{
    class Mat;

    namespace cuda
    {
        class GpuMat;
    }
}

//
namespace DataStructures
{
    struct CUDAImage;
}

/**
 * @namespace Decoding
 *
 * @brief
 */
namespace Decoding
{

/**
 * @class IImageDecoder
 *
 * @brief
 */
class IImageDecoder
{
public:

    /**
     * @brief
     */
    IImageDecoder() = default;

    /**
     * @brief
     */
    virtual ~IImageDecoder() = default;

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param decodedImage
     */
    virtual void Decode(const unsigned char* data, unsigned long long size, cv::Mat& decodedImage) = 0;

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param decodedImage
     */
    virtual void Decode(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& decodedImage) = 0;

    /**
     * @brief
     *
     * @param data
     * @param size
     * @param decodedImage
     */
    virtual void Decode(const unsigned char* data, unsigned long long size, DataStructures::CUDAImage& decodedImage) = 0;

    /**
     * @brief
     */
    virtual void Initialize() = 0;

    /**
     * @brief
     *
     * @return
     */
    virtual bool IsInitialized() = 0;

protected:

    /**
     * @brief
     *
     * @param width
     * @param height
     * @param channels
     */
    virtual void AllocateBuffer(int width, int height, int channels) = 0;

};

}

#endif // INTERFACE_IMAGE_DECODER_H