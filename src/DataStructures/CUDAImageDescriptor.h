#ifndef CUDA_IMAGE_DESCRIPTOR_H
#define CUDA_IMAGE_DESCRIPTOR_H

#include "CUDAImage.h"

/**
 * @namespace DataStructures
 *
 * @brief
 */
namespace DataStructures
{

/**
 * @class CUDAImageDescriptor
 *
 * @brief
 */
class CUDAImageDescriptor final
{
public:

private:

    ///
    CUDAImage image_;

    ///
    int frameId_;

    ///
    int cameraId_;

    ///
    int timestamp_;
};

}

#endif // CUDA_IMAGE_DESCRIPTOR_H
