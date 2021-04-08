/**
 * @file IImageEncoder.h.
 *
 * @brief
 */

#ifndef INTERFACE_IMAGE_ENCODER_H
#define INTERFACE_IMAGE_ENCODER_H

#include <vector>

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
 * @namespace Encoding
 *
 * @brief
 */
namespace Encoding
{

/**
 * @class IImageEncoder
 *
 * @brief
 */
class IImageEncoder
{

public:

    IImageEncoder() = default;

    virtual ~IImageEncoder() = default;

    void EncodeImage(DataStructures::CUDAImage& image, std::vector<unsigned char> output);

private:

};

}


#endif // INTERFACE_IMAGE_ENCODER_H
