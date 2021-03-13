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
    CUDAImageDescriptor();

    ~CUDAImageDescriptor();

    bool operator==(const CUDAImageDescriptor& other);

    CUDAImageDescriptor& operator=(const CUDAImageDescriptor& other);

    CUDAImageDescriptor& operator=(CUDAImageDescriptor&& other) noexcept;

    const CUDAImage& GetCUDAImage();

    void SetCUDAImage(const CUDAImage& image);

    void SetCUDAImage(CUDAImage&& image) noexcept;

    int GetFrameId() noexcept;

    void SetFrameId(int frameId);

    int GetCameraId() noexcept;

    void SetCameraId(int cameraId);

    int GetTimestamp() noexcept;

    void SetTimestamp(int timestamp);


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
