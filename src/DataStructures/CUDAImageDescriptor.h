/**
 * @file CUDAImageDescriptor.h.
 *
 * @brief Declares class for containing CUDA image and its metadata.
 */


#ifndef CUDA_IMAGE_DESCRIPTOR_H
#define CUDA_IMAGE_DESCRIPTOR_H

#include <vector>

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
     *
     * @param other
     */
    CUDAImageDescriptor(const CUDAImageDescriptor& other) = default;

    /**
     * @brief
     *
     * @param other
     */
    CUDAImageDescriptor(CUDAImageDescriptor&& other) noexcept;

    /**
     * @brief
     */
    ~CUDAImageDescriptor() = default;

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
    CUDAImageDescriptor& operator=(const CUDAImageDescriptor& other) = default;

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
    [[nodiscard]] int GetFrameId() const noexcept;

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
    int GetCameraId() const noexcept;

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
    [[nodiscard]] unsigned long GetTimestamp() const noexcept;

    /**
     * @brief
     *
     * @param timestamp
     */
    void SetTimestamp(unsigned long timestamp);

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] float GetFocalLength() const noexcept;

    /**
     * @brief
     *
     * @param focalLength
     */
    void SetFocalLength(float focalLength);

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] float GetSensorSize() const noexcept;

    /**
     * @brief
     *
     * @param sensorSize
     */
    void SetSensorSize(float sensorSize);

    /**
     * @brief
     *
     * @return
     */
    const std::vector<unsigned char>& GetRawImageData() noexcept;

    /**
     * @brief
     *
     * @param rawImageData
     */
    void SetRawImageData(const std::vector<unsigned char>& rawImageData);

    /**
     * @brief
     *
     * @param rawImageData
     */
    void SetRawImageData(std::vector<unsigned char>&& rawImageData) noexcept;

    /**
     * @brief
     */
    void ClearRawImageData() noexcept;

private:

    ///
    CUDAImage image_;

    ///
    int frameId_;

    ///
    int cameraId_;

    ///
    unsigned long timestamp_;

    ///
    float focalLength_;

    ///
    float sensorSize_;

    ///
    std::vector<unsigned char> rawImageData_;
};

}

#endif // CUDA_IMAGE_DESCRIPTOR_H
