/**
 * @file CUDAImageDescriptor.h.
 *
 * @brief Declares class for containing CUDA image and its metadata.
 */


#ifndef CUDA_IMAGE_DESCRIPTOR_H
#define CUDA_IMAGE_DESCRIPTOR_H

#include "CUDAImage.h"

/**
 * @namespace DataStructures
 *
 * @brief Namespace of libdatastructures library.
 */
namespace DataStructures
{

/**
 * @class CUDAImageDescriptor
 *
 * @brief This is a class for containing CUDA image and its metadata.
 */
class CUDAImageDescriptor final
{
public:

    /**
     * @brief
     */
    CUDAImageDescriptor();

    /**
     * @brief
     */
    ~CUDAImageDescriptor();

    /**
     * @brief
     *
     * @param other
     * @return
     */
    bool operator==(const CUDAImageDescriptor& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImageDescriptor& operator=(const CUDAImageDescriptor& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImageDescriptor& operator=(CUDAImageDescriptor&& other) noexcept;

    /**
     * @brief
     *
     * @return
     */
    const CUDAImage& GetCUDAImage();

    /**
     * @brief
     *
     * @param image
     */
    void SetCUDAImage(const CUDAImage& image);

    /**
     * @brief
     *
     * @param image
     */
    void SetCUDAImage(CUDAImage&& image) noexcept;

    /**
     * @brief
     *
     * @return
     */
    int GetFrameId() noexcept;

    /**
     * @brief
     *
     * @param frameId
     */
    void SetFrameId(int frameId);

    /**
     * @brief
     *
     * @return
     */
    int GetCameraId() noexcept;

    /**
     * @brief
     *
     * @param cameraId
     */
    void SetCameraId(int cameraId);

    /**
     * @brief
     *
     * @return
     */
    int GetTimestamp() noexcept;

    /**
     * @brief
     *
     * @param timestamp
     */
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

    float focalLength_;

    float sensorSize_;
};

}

#endif // CUDA_IMAGE_DESCRIPTOR_H
