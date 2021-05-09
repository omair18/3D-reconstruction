/**
 * @file CUDAImage.h.
 *
 * @brief Declares struct for working with images using CUDA.
 */


#ifndef CUDA_IMAGE_H
#define CUDA_IMAGE_H

#include <cstddef>

// forward declaration for cv::Mat and cv::cuda::GpuMat
namespace cv
{
    class Mat;
    namespace cuda
    {
        class GpuMat;
    }
}

/**
 * @namespace DataStructures
 *
 * @brief Namespace of libdatastructures library.
 */
namespace DataStructures
{

/**
 * @struct CUDAImage
 *
 * @brief This is a struct for working with images using CUDA.
 */
struct CUDAImage final
{

    /**
     * @enum ELEMENT_TYPE
     *
     * @brief Declares CUDA image single element types.
     */
    enum ELEMENT_TYPE
    {
        /// 8-bit unsigned element
        TYPE_8U = 0,

        /// 8-bit signed element
        TYPE_8S,

        /// 16-bit unsigned element
        TYPE_16U,

        /// 16-bit signed element
        TYPE_16S,

        /// 16-bit floating point element
        TYPE_16F,

        /// 32-bit signed element
        TYPE_32S,

        /// 32-bit floating point element
        TYPE_32F,

        /// 64-bit floating point element
        TYPE_64F,

        /// Unknown element type
        TYPE_UNKNOWN
    };

    /**
     * @brief Default constructor.
     */
    CUDAImage();

    /**
     * @brief Copy constructor.
     *
     * @param other - Image to copy
     */
    CUDAImage(const CUDAImage& other);

    /**
     * @brief Move constructor.
     *
     * @param other - Image to move
     */
    CUDAImage(CUDAImage&& other) noexcept;

    /**
     * @brief Destructor.
     */
    ~CUDAImage();

    /**
     * @brief Equation operator. Copies data from other-params CUDA image to current image. If current image didn't have
     * enough memory for copying data, performs memory reallocation.
     *
     * @param other - Image to be copied
     * @return L-value reference on current image.
     */
    CUDAImage& operator=(const CUDAImage& other);

    /**
     * @brief Equation operator. Moves data from other-params CUDA image to current image.
     *
     * @param other - Image to be moved
     * @return L-value reference on current image.
     */
    CUDAImage& operator=(CUDAImage&& other) noexcept(false);

    /**
     * @brief Equation operator. Copies data from other-params cv::cuda::GpuMat to current CUDA image.
     * If current image doesn't have enough memory, performs reallocation of current image.
     *
     * @param other - Image to be copied
     * @return L-value reference on current image.
     */
    CUDAImage& operator=(const cv::cuda::GpuMat& other);

    /**
     * @brief Equation operator. Moves data from other-params cv::cuda::GpuMat image to current image.
     *
     * @param other - Image to be moved
     * @return L-value reference on current image.
     */
    CUDAImage& operator=(cv::cuda::GpuMat&& other);

    /**
     * @brief Equation operator. Copies data from other-params cv::Mat to current CUDA image.
     * If current image doesn't have enough memory, performs reallocation of current image.
     *
     * @param other - Image to be copied
     * @return L-value reference on current image.
     */
    CUDAImage& operator=(const cv::Mat& other);

    /**
     * @brief Equation operator. Moves data from other-params cv::Mat image to current image.
     *
     * @note As it's impossible to move host data to device, performs copy operation.
     *
     * @param other - Image to be moved
     * @return L-value reference on current image.
     */
    CUDAImage& operator=(cv::Mat&& other) noexcept;

    /**
     * @brief Allocates memory on GPU for image with given parameters in synchronous way. If memory was already
     * allocated, then old memory chunk will be released also in synchronous way.
     *
     * @param width - Width of the image in pixels
     * @param height - Height of the image in pixels
     * @param channels - Amount of channels of the image
     * @param type - Type of single element of image
     * @param pitchedAllocation - Flag of pitched allocation
     */
    void Allocate(unsigned int width, unsigned int height, unsigned int channels, ELEMENT_TYPE type, bool pitchedAllocation);

    /**
     * @brief Allocates memory on GPU for image with given parameters in asynchronous way. If memory was already
     * allocated, then old memory chunk will be released also in synchronous way.
     *
     * @note CUDA API doesn't have cudaMallocPitchAsync, so this kind of memory allocation will always be synchronous.
     *
     * @param width - Width of the image in pixels
     * @param height - Height of the image in pixels
     * @param channels - Amount of channels of the image
     * @param type - Type of single element of image
     * @param pitchedAllocation - Flag of pitched allocation
     * @param cudaStream - CUDA stream to perform asynchronous memory allocation
     */
    void AllocateAsync(unsigned int width, unsigned int height, unsigned int channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream = nullptr);

    /**
     * @brief Releases GPU memory of current image in synchronous way.
     */
    void Release();

    /**
     * @brief Releases GPU memory of current image in asynchronous way.
     *
     * @param cudaStream - CUDA stream to perform asynchronous memory releasing
     */
    void ReleaseAsync(void* cudaStream = nullptr);

    /// CUDA image

    /**
     * @brief Moves data from current CUDA image to dst-param image. If dst-param image already has allocated data,
     * then performs memory releasing in synchronous way.
     *
     * @param dst - Image where data from current image will be moved
     */
    void MoveToCUDAImage(CUDAImage& dst);

    /**
     * @brief Moves data from current CUDA image to dst-param image. If dst-param image already has allocated data,
     * then performs memory releasing in asynchronous way.
     *
     * @param dst - Image where data from current image will be moved
     * @param cudaStream - CUDA stream for performing asynchronous memory releasing
     */
    void MoveToCUDAImageAsync(CUDAImage& dst, void* cudaStream = nullptr);

    /**
     * @brief Copies data from current CUDA image to dst-param image. If dst-param image already has allocated data,
     * checks if there is enough memory on destination-side image to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs synchronous memory reallocation and copies data from current image to allocated
     * memory in synchronous way.
     *
     * @param dst - Image where data from current image will be copied
     */
    void CopyToCUDAImage(CUDAImage& dst) const;

    /**
     * @brief Copies data from current CUDA image to dst-param image. If dst-param image already has allocated data,
     * checks if there is enough memory on destination-side image to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs asynchronous memory reallocation and copies data from current image to allocated
     * memory in asynchronous way.
     *
     * @param dst - Image where data from current image will be copied
     * @param cudaStream - CUDA stream for performing asynchronous operations copy and memory reallocation operations
     */
    void CopyToCUDAImageAsync(CUDAImage& dst, void* cudaStream = nullptr) const;

    /**
     * @brief Moves data from src-param CUDA image to current image. If current image already has allocated data,
     * then performs memory releasing in synchronous way.
     *
     * @param src - Image to move
     */
    void MoveFromCUDAImage(CUDAImage& src);

    /**
     * @brief Moves data from src-param CUDA image to current image. If current image already has allocated data,
     * then performs memory releasing in asynchronous way.
     *
     * @param src - Image to move
     * @param cudaStream - CUDA stream for asynchronous memory releasing operation
     */
    void MoveFromCUDAImageAsync(CUDAImage& src, void* cudaStream = nullptr);

    /**
     * @brief Copies data from src-param CUDA image to current image. If current image already has allocated data,
     * checks if current image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs synchronous memory reallocation and copies data from src-param image to
     * allocated memory in synchronous way.
     *
     * @param src - Image to copy
     */
    void CopyFromCUDAImage(const CUDAImage& src);

    /**
     * @brief Copies data from src-param CUDA image to current image. If current image already has allocated data,
     * checks if current image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs synchronous memory reallocation and copies data from src-param image to
     * allocated memory in synchronous way.
     *
     * @param src - Image to copy
     * @param cudaStream - CUDA stream for performing asynchronous operations copy and memory reallocation operations
     */
    void CopyFromCUDAImageAsync(const CUDAImage& src, void* cudaStream = nullptr);

    /// Cv GPU Mat

    /**
     * @brief Moves data from current CUDA image to dst-param image. If dst-param image already has allocated memory -
     * performs memory releasing operation.
     *
     * @param dst - Image where data from current image will be moved
     */
    void MoveToGpuMat(cv::cuda::GpuMat& dst);

    /**
     * @brief Moves data from current CUDA image to dst-param image. If dst-param image already has allocated data,
     * then performs memory releasing in asynchronous way.
     *
     * @param dst - Image where data from current image will be moved
     * @param cudaStream - CUDA stream for performing asynchronous memory releasing
     */
    void MoveToGpuMatAsync(cv::cuda::GpuMat& dst, void* cudaStream = nullptr);

    /**
     * @brief Copies data from current image to dst-param image. If dst-param image already has allocated data,
     * checks if dst-param image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs synchronous memory reallocation and copies data from current image to
     * allocated memory in synchronous way.
     *
     * @param dst - Image where data from current image will be copied
     */
    void CopyToGpuMat(cv::cuda::GpuMat& dst) const;

    /**
     * @brief Copies data from current image to dst-param image. If dst-param image already has allocated data,
     * checks if dst-param image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs synchronous memory reallocation and copies data from current image to
     * allocated memory in asynchronous way.
     *
     * @param dst - Image where data from current image will be copied
     * @param cudaStream - CUDA stream for performing asynchronous copy operation
     */
    void CopyToGpuMatAsync(cv::cuda::GpuMat& dst, void* cudaStream = nullptr) const;

    /**
     * @brief Moves data from src-param image to current image. If current image already has allocated data,
     * then performs memory releasing in synchronous way.
     *
     * @param src - Image to move
     */
    void MoveFromGpuMat(cv::cuda::GpuMat& src);

    /**
     * @brief Moves data from src-param image to current image. If current image already has allocated data,
     * then performs memory releasing in asynchronous way.
     *
     * @param src - Image to move
     * @param cudaStream - CUDA stream for performing asynchronous memory releasing operation
     */
    void MoveFromGpuMatAsync(cv::cuda::GpuMat& src, void* cudaStream = nullptr);

    /**
     * @brief Copies data from src-param image to current image. If current image already has allocated data,
     * checks if current image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs synchronous memory reallocation and copies data from src-param image to
     * allocated memory in synchronous way.
     *
     * @param src - Image to copy
     */
    void CopyFromGpuMat(const cv::cuda::GpuMat& src);

    /**
     * @brief Copies data from src-param image to current image. If current image already has allocated data,
     * checks if current image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs asynchronous memory reallocation and copies data from src-param image to
     * allocated memory in asynchronous way.
     *
     * @param src - Image to copy
     * @param cudaStream - CUDA stream for performing asynchronous memory allocation and copy operations
     */
    void CopyFromGpuMatAsync(const cv::cuda::GpuMat& src, void* cudaStream = nullptr);

    /// Cv Mat

    /**
     * @brief Moves data from current CUDA image to dst-param image. Reallocates dst-param image in synchronous way
     * and copies data from current image to dst-param image also in synchronous way, then releases data of current
     * image in synchronous way.
     *
     * @param dst - Image where data from current image will be moved
     */
    void MoveToCvMat(cv::Mat& dst);

    /**
     * @brief Moves data from current CUDA image to dst-param image. Reallocates dst-param image in synchronous way
     * and copies data from current image to dst-param image in asynchronous way, then releases data of current
     * image in asynchronous way.
     *
     * @param dst - Image where data from current image will be moved
     * @param cudaStream - CUDA stream for performing asynchronous memory releasing and copy operations
     */
    void MoveToCvMatAsync(cv::Mat& dst, void* cudaStream = nullptr);

    /**
     * @brief Copies data from current CUDA image to dst-param image. Reallocates dst-param image in synchronous way
     * and copies data from current image to dst-param image also in synchronous way.
     *
     * @param dst - Image where data from current image will be copied
     */
    void CopyToCvMat(cv::Mat& dst) const;

    /**
     * @brief Copies data from current CUDA image to dst-param image. Reallocates dst-param image in synchronous way
     * and copies data from current image to dst-param image in asynchronous way.
     *
     * @param dst - Image where data from current image will be copied
     * @param cudaStream - CUDA stream for performing asynchronous copy operation
     */
    void CopyToCvMatAsync(cv::Mat& dst, void* cudaStream = nullptr) const;

    /**
     * @brief Copies data from src-param image using CopyFromCvMat method, then clears data from src-param image.
     *
     * @param src - Image to move
     */
    void MoveFromCvMat(cv::Mat& src);

    /**
     * @brief Copies data from src-param image using CopyFromCvMatAsync method, then clears data from src-param image.
     *
     * @param src - Image to move
     * @param cudaStream - CUDA stream for performing asynchronous copy operation
     */
    void MoveFromCvMatAsync(cv::Mat& src, void* cudaStream = nullptr);

    /**
     * @brief Copies data from src-param image to current image. If current image already has allocated data,
     * checks if current image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs synchronous memory reallocation and copies data from src-param image to
     * allocated memory in synchronous way.
     *
     * @param src - Image to copy
     */
    void CopyFromCvMat(const cv::Mat& src);

    /**
     * @brief Copies data from src-param image to current image. If current image already has allocated data,
     * checks if current image has enough memory to perform copy operation. If there is enough memory,
     * then just performs copy operation and resets image's parameters to the parameters of copied image. If there is
     * not enough memory, then performs asynchronous memory reallocation and copies data from src-param image to
     * allocated memory in asynchronous way.
     *
     * @param src - Image to copy
     * @param cudaStream - CUDA stream for performing asynchronous memory releasing and copy operations
     */
    void CopyFromCvMatAsync(const cv::Mat& src, void* cudaStream = nullptr);

    /// Raw host pointer

    /**
     * @brief Copies data from src-param host raw pointer to current image in asynchronous way. If current image
     * already has allocated GPU memory, then checks if data from src-param can be copied to current image. If there is
     * enough memory to perform copy operation the performs it. In other case reallocates memory with
     * pitchedAllocation-param flag.
     *
     * @note May be unsafe.
     *
     * @param src - Raw host pointer initialized with address of some host memory chunk
     * @param width - Width of image pointed by src-param
     * @param height - Height of image pointed by src-param
     * @param channels - Amount of channel of image pointed by src-param
     * @param type - Type of single element of image pointed by dst-param
     * @param pitchedAllocation - Flag of memory allocation mode. If this flag is true - using pitched memory
     */
    void CopyFromRawHostPointer(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation);

    /**
     * @brief Copies data from src-param host raw pointer to current image in asynchronous way. If current image
     * already has allocated GPU memory, then checks if data from src-param can be copied to current image. If there is
     * enough memory to perform copy operation the performs it. In other case reallocates memory with
     * pitchedAllocation-param flag.
     *
     * @note May be unsafe.
     *
     * @param src - Raw host pointer initialized with address of some host memory chunk
     * @param width - Width of image pointed by src-param
     * @param height - Height of image pointed by src-param
     * @param channels - Amount of channel of image pointed by src-param
     * @param type - Type of single element of image pointed by dst-param
     * @param pitchedAllocation - Flag of memory allocation mode. If this flag is true - using pitched memory
     * @param cudaStream - CUDA stream for performing asynchronous memory allocation and copy operations
     */
    void CopyFromRawHostPointerAsync(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream = nullptr);

    /**
     * @brief Copies data from current CUDA image to raw host pointer in synchronous way.
     *
     * @note May be unsafe.
     *
     * @param dst - Raw device pointer initialized with address of some GPU memory chunk
     * @param width - Width of image pointed by dst-param
     * @param channels - Amount of channel of image pointed by dst-param
     * @param type - Type of single element of image pointed by dst-param
     */
    void CopyToRawHostPointer(void* dst, size_t width, size_t channels, ELEMENT_TYPE type) const;

    /**
     * @brief Copies data from current CUDA image to raw device pointer in asynchronous way.
     *
     * @note May be unsafe.
     *
     * @param dst - Raw device pointer initialized with address of some GPU memory chunk
     * @param width - Width of image pointed by dst-param
     * @param channels - Amount of channel of image pointed by dst-param
     * @param type - Type of single element of image pointed by dst-param
     * @param cudaStream - CUDA stream for performing asynchronous copy operation
     */
    void CopyToRawHostPointerAsync(void* dst, size_t width, size_t channels, ELEMENT_TYPE type, void* cudaStream = nullptr) const;

    /// Device pointer

    /**
     * @brief Copies data from src-param device raw pointer to current image in synchronous way. If current image
     * already has allocated GPU memory, then checks if data from src-param can be copied to current image. If there is
     * enough memory to perform copy operation the performs it. In other case reallocates memory with
     * pitchedAllocation-param flag.
     *
     * @note May be unsafe.
     *
     * @param src - Raw device pointer initialized with address of some GPU memory chunk
     * @param width - Width of image pointed by src-param
     * @param height - Height of image pointed by src-param
     * @param channels - Amount of channel of image pointed by src-param
     * @param type - Type of single element of image pointed by dst-param
     * @param pitchedAllocation - Flag of memory allocation mode. If this flag is true - using pitched memory
     * allocation, in other case allocates simple linear memory chunk
     */
    void CopyFromRawDevicePointer(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation);

    /**
     * @brief Copies data from src-param device raw pointer to current image in asynchronous way. If current image
     * already has allocated GPU memory, then checks if data from src-param can be copied to current image. If there is
     * enough memory to perform copy operation the performs it. In other case reallocates memory with
     * pitchedAllocation-param flag.
     *
     * @note May be unsafe.
     *
     * @param src - Raw device pointer initialized with address of some GPU memory chunk
     * @param width - Width of image pointed by src-param
     * @param height - Height of image pointed by src-param
     * @param channels - Amount of channel of image pointed by src-param
     * @param type - Type of single element of image pointed by dst-param
     * @param pitchedAllocation - Flag of memory allocation mode. If this flag is true - using pitched memory
     * allocation, in other case allocates simple linear memory chunk
     * @param cudaStream - CUDA stream for performing asynchronous copy and memory allocation operations
     */
    void CopyFromRawDevicePointerAsync(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream = nullptr);

    /**
     * @brief Copies data from current CUDA image to raw device pointer in synchronous way.
     *
     * @note May be unsafe.
     *
     * @param dst - Raw device pointer initialized with address of some GPU memory chunk
     * @param width - Width of image pointed by dst-param
     * @param channels - Amount of channel of image pointed by dst-param
     * @param type - Type of single element of image pointed by dst-param
     */
    void CopyToRawDevicePointer(void* dst, size_t width, size_t channels, ELEMENT_TYPE type) const;

    /**
     * @brief Copies data from current CUDA image to raw device pointer in asynchronous way.
     *
     * @note May be unsafe.
     *
     * @param dst - Raw device pointer initialized with address of some GPU memory chunk
     * @param width - Width of image pointed by dst-param
     * @param channels - Amount of channel of image pointed by dst-param
     * @param type - Type of single element of image pointed by dst-param
     * @param cudaStream - CUDA stream for performing asynchronous copy operation
     */
    void CopyToRawDevicePointerAsync(void* dst, size_t width, size_t channels, ELEMENT_TYPE type, void* cudaStream = nullptr) const;

    /**
     * @brief Checks, weather memory on GPU was allocated for this image.
     *
     * @return True if image has a pointer to GPU data. Otherwise returns false.
     */
    [[nodiscard]] bool Allocated() const;

    /**
     * @brief Returns size of single image's element.
     *
     * @return Size of single image's element.
     */
    [[nodiscard]] size_t GetElementSize() const;

    /// Width of the image.
    size_t width_ = 0;

    /// Height of the image.
    size_t height_ = 0;

    /// Pitch size in bytes.
    size_t pitch_ = 0;

    /// Amount of image's channels.
    size_t channels_ = 0;

    /// Size of allocated device memory chunk for image.
    size_t allocatedBytes_ = 0;

    /// Type of single element of image
    ELEMENT_TYPE elementType_ = ELEMENT_TYPE::TYPE_UNKNOWN;

    /// Device pointer to image's data
    unsigned char* gpuData_ = nullptr;

    /// Flag of pitched allocation.
    bool pitchedAllocation_ = false;
};

}

#endif // CUDA_IMAGE_H
