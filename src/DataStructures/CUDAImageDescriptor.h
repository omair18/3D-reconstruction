/**
 * @file CUDAImageDescriptor.h.
 *
 * @brief Declares the class for containing CUDA image and its metadata.
 */


#ifndef CUDA_IMAGE_DESCRIPTOR_H
#define CUDA_IMAGE_DESCRIPTOR_H

#include <vector>
#include <memory>

/**
 * @namespace DataStructures
 *
 * @brief Namespace of libdatastructures library.
 */
namespace DataStructures
{

// forward declaration for DataStructures::CUDAImage
class CUDAImage;

/**
 * @class CUDAImageDescriptor
 *
 * @brief This is a class for containing CUDA image and its metadata.
 */
class CUDAImageDescriptor final
{
public:

    /**
     * @brief Default constructor.
     */
    CUDAImageDescriptor();

    /**
     * @brief Copy constructor.
     *
     * @param other - CUDA image descriptor to copy
     */
    CUDAImageDescriptor(const CUDAImageDescriptor& other);

    /**
     * @brief Move constructor.
     *
     * @param other - CUDA image desctiptor to move
     */
    CUDAImageDescriptor(CUDAImageDescriptor&& other) noexcept;

    /**
     * @brief Default destructor.
     */
    ~CUDAImageDescriptor();

    /**
     * @brief Checks weather current image descriptor and other-param image descriptor have same pointers on CUDA
     * images.
     *
     * @param other - Descriptor to compare with
     * @return True if pointers are initialized with same address. Otherwise returns false.
     */
    bool operator==(const CUDAImageDescriptor& other);

    /**
     * @brief Equation operator. Copies data from other-param CUDA image descriptor to current image descriptor.
     *
     * @param other - Image descriptor to copy data from
     * @return L-value reference to current CUDA image descriptor.
     */
    CUDAImageDescriptor& operator=(const CUDAImageDescriptor& other);

    /**
     * @brief Equation operator. Moves data from other-param CUDA image descriptor to current image descriptor.
     *
     * @param other - Image descriptor to move data from
     * @return L-value reference to current CUDA image descriptor.
     */
    CUDAImageDescriptor& operator=(CUDAImageDescriptor&& other) noexcept;

    /**
     * @brief Provides access to image_-member.
     *
     * @return Pointer to current descriptor's CUDA image
     */
    [[nodiscard]] const std::unique_ptr<CUDAImage>& GetCUDAImage() const;

    /**
     * @brief Copies data of image-param CUDA image to the image pointed by image_-member.
     *
     * @param image - CUDA image for being copied
     */
    void SetCUDAImage(const CUDAImage& image);

    /**
     * @brief Moves data of image-param CUDA image to the image pointed by image_-member.
     *
     * @param image - CUDA image for being moved
     */
    void SetCUDAImage(CUDAImage&& image) noexcept;

    /**
     * @brief Copies data from CUDA image pointed by image-param to CUDA image pointed by image_-member.
     *
     * @param image - Pointer to CUDA image to be copied
     */
    void SetCUDAImage(const std::unique_ptr<CUDAImage>& image);

    /**
     * @brief Moves image-param pointer to image_member.
     *
     * @param image - Pointer to CUDA image.
     */
    void SetCUDAImage(std::unique_ptr<CUDAImage>&& image) noexcept;

    /**
     * @brief Provides access to frameId_-member.
     *
     * @return Value of frameId_-member
     */
    [[nodiscard]] int GetFrameId() const noexcept;

    /**
     * @brief Sets value of frameId_-member to frameId-param.
     *
     * @param frameId - Value of frameId_-member
     */
    void SetFrameId(int frameId);

    /**
     * @brief Provides access to cameraId_-member.
     *
     * @return Value of cameraId_-member
     */
    [[nodiscard]] int GetCameraId() const noexcept;

    /**
     * @brief Sets value of cameraId_-member to cameraId-param.
     *
     * @param cameraId - Value of cameraId_-member
     */
    void SetCameraId(int cameraId);

    /**
     * @brief Provides access to timestamp_-member.
     *
     * @return Value of timestamp_-member
     */
    [[nodiscard]] unsigned long GetTimestamp() const noexcept;

    /**
     * @brief Sets value of timestamp_-member to timestamp-param.
     *
     * @param timestamp - Value of timestamp_-member
     */
    void SetTimestamp(unsigned long timestamp);

    /**
     * @brief Provides access to focalLength_-member.
     *
     * @return Value of focalLength_-member
     */
    [[nodiscard]] float GetFocalLength() const noexcept;

    /**
     * @brief Sets value of camera's focal length to focalLength-param.
     *
     * @param focalLength - Value of camera's focal length in millimeters
     */
    void SetFocalLength(float focalLength);

    /**
     * @brief Provides access to sensorSize_-member.
     *
     * @return Value of sensorSize_-member
     */
    [[nodiscard]] float GetSensorSize() const noexcept;

    /**
     * @brief Sets value of camera's sensor size to sensorSize-param.
     *
     * @param sensorSize - Value of camera's sensor size in millimeters
     */
    void SetSensorSize(float sensorSize);

    /**
     * @brief Provides access to rawImageData_-member.
     *
     * @return L-value reference to the rawImageData_-member
     */
    [[nodiscard]] const std::vector<unsigned char>& GetRawImageData() const noexcept;

    /**
     * @brief Copies content of rawImageData-param to rawImageData_-member.
     *
     * @param rawImageData - Data to copy to rawImageData_-member
     */
    void SetRawImageData(const std::vector<unsigned char>& rawImageData);

    /**
     * @brief Moves content of rawImageData-param to rawImageData_-member.
     *
     * @param rawImageData - Data to move to rawImageData_-member
     */
    void SetRawImageData(std::vector<unsigned char>&& rawImageData) noexcept;

    /**
     * @brief Clears the content of rawImageData_-member.
     */
    void ClearRawImageData() noexcept;

private:

    /// Pointer to CUDA image.
    std::unique_ptr<CUDAImage> image_;

    /// ID of frame in dataset.
    int frameId_;

    /// ID of dataset's camera.
    int cameraId_;

    /// Timestamp of receiving image's data.
    unsigned long timestamp_;

    /// Focal length of the dataset's camera in millimeters.
    float focalLength_;

    /// Sensor size of the dataset's camera in millimeters.
    float sensorSize_;

    /// Raw image data for image encoding and decoding algorithms.
    std::vector<unsigned char> rawImageData_;
};

}

#endif // CUDA_IMAGE_DESCRIPTOR_H
