/**
 * @file ImageDescriptor.h.
 *
 * @brief Declares the class for containing image and its metadata.
 */


#ifndef IMAGE_DESCRIPTOR_H
#define IMAGE_DESCRIPTOR_H

#include <vector>
#include <memory>

// forward declaration for cv::Mat
namespace cv
{
    class Mat;
}

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
 * @class ImageDescriptor
 *
 * @brief This is a class for containing image and its metadata.
 */
class ImageDescriptor final
{
public:

    /**
     * @enum LOCATION
     *
     * @brief Possible variants of descriptor's data location.
     */
    enum LOCATION
    {
        /// Descriptor's data is located on host memory
        HOST = 0,

        /// Descriptor's data is located on device memory
        DEVICE,

        /// Descriptor's data location is undefined
        UNDEFINED
    };

    /**
     * @brief Default constructor.
     */
    ImageDescriptor();

    /**
     * @brief Copy constructor.
     *
     * @param other - CUDA image descriptor to copy
     */
    ImageDescriptor(const ImageDescriptor& other);

    /**
     * @brief Move constructor.
     *
     * @param other - CUDA image desctiptor to move
     */
    ImageDescriptor(ImageDescriptor&& other) noexcept;

    /**
     * @brief Default destructor.
     */
    ~ImageDescriptor();

    /**
     * @brief Equation operator. Copies data from other-param CUDA image descriptor to current image descriptor.
     *
     * @param other - Image descriptor to copy data from
     * @return L-value reference to current CUDA image descriptor.
     */
    ImageDescriptor& operator=(const ImageDescriptor& other);

    /**
     * @brief Equation operator. Moves data from other-param CUDA image descriptor to current image descriptor.
     *
     * @param other - Image descriptor to move data from
     * @return L-value reference to current CUDA image descriptor.
     */
    ImageDescriptor& operator=(ImageDescriptor&& other) noexcept;

    /**
     * @brief Provides access to CUDAImage_-member.
     *
     * @return Pointer to current descriptor's CUDA image
     */
    [[nodiscard]] const std::unique_ptr<CUDAImage>& GetCUDAImage() const;

    /**
     * @brief Copies data of image-param CUDA image to the image pointed by CUDAImage_-member.
     *
     * @param image - CUDA image for being copied
     */
    void SetCUDAImage(const CUDAImage& image);

    /**
     * @brief Moves data of image-param CUDA image to the image pointed by CUDAImage_-member.
     *
     * @param image - CUDA image for being moved
     */
    void SetCUDAImage(CUDAImage&& image) noexcept;

    /**
     * @brief Copies data from CUDA image pointed by image-param to CUDA image pointed by CUDAImage_-member.
     *
     * @param image - Pointer to CUDA image to be copied
     */
    void SetCUDAImage(const std::unique_ptr<CUDAImage>& image);

    /**
     * @brief Moves image-param pointer to CUDAImage_-member.
     *
     * @param image - Pointer to CUDA image
     */
    void SetCUDAImage(std::unique_ptr<CUDAImage>&& image) noexcept;

    /**
     * @brief Copies data of image-param host image to the image pointed by hostImage_-member.
     *
     * @param image - Host image for being copied
     */
    void SetHostImage(const cv::Mat& image);

    /**
     * @brief Moves data of image-param host image to the image pointed by hostImage_-member.
     *
     * @param image - Host image for being moved
     */
    void SetHostImage(cv::Mat&& image);

    /**
     * @brief Copies data from host image pointed by image-param to host image pointed by hostImage_-member.
     *
     * @param image - Pointer to host image to be copied
     */
    void SetHostImage(const std::unique_ptr<cv::Mat>& image);

    /**
     * @brief Moves image-param pointer to hostImage_-member.
     *
     * @param image - Pointer to host image
     */
    void SetHostImage(std::unique_ptr<cv::Mat>&& image);

    /**
     * @brief Provides access to hostImage_-member.
     *
     * @return L-value reference to a hostImage_-member
     */
    [[nodiscard]] const std::unique_ptr<cv::Mat>& GetHostImage() const noexcept;

    /**
     * @brief Sets location_-member flag to a location-param value.
     *
     * @param location - New value of location_-member flag.
     */
    void SetDataLocation(LOCATION location);

    /**
     * @brief Provides access to location_-member.
     */
    [[nodiscard]] LOCATION GetDataLocation() const noexcept;

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

    /**
     * @brief Sets value of distortionFunctionId_-member to id-param value.
     *
     * @param id - New value of distortionFunctionId_-member
     */
    void SetCameraDistortionFunctionId(int id);

    /**
     * @brief Provides access to distortionFunctionId_-member.
     *
     * @return Value of distortionFunctionId_-member
     */
    [[nodiscard]] int GetCameraDistortionFunctionId() const noexcept;

private:

    /// Pointer to CUDA image.
    std::unique_ptr<CUDAImage> CUDAImage_;

    /// Pointer to host image.
    std::unique_ptr<cv::Mat> hostImage_;

    /// Data location flag.
    LOCATION location_;

    /// ID of frame in dataset.
    int frameId_;

    /// ID of dataset's camera.
    int cameraId_;

    /// ID of camera's distortion function.
    int distortionFunctionId_;

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

#endif // IMAGE_DESCRIPTOR_H
