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
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImage& operator=(const CUDAImage& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImage& operator=(CUDAImage&& other) noexcept(false);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImage& operator=(const cv::cuda::GpuMat& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImage& operator=(cv::cuda::GpuMat&& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImage& operator=(const cv::Mat& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    CUDAImage& operator=(cv::Mat&& other) noexcept;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    bool operator==(const CUDAImage& other) const;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    bool operator==(const cv::cuda::GpuMat& other) const;

    /**
     * @brief
     *
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param pitchedAllocation
     */
    void Allocate(unsigned int width, unsigned int height, unsigned int channels, ELEMENT_TYPE type, bool pitchedAllocation);

    /**
     * @brief
     *
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param pitchedAllocation
     * @param cudaStream
     */
    void AllocateAsync(unsigned int width, unsigned int height, unsigned int channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream = nullptr);

    /**
     * @brief
     */
    void Release();

    /**
     * @brief
     *
     * @param cudaStream
     */
    void ReleaseAsync(void* cudaStream = nullptr);

    /// CUDA image

    /**
     * @brief
     *
     * @param dst
     */
    void MoveToCUDAImage(CUDAImage& dst);

    /**
     * @brief
     *
     * @param dst
     * @param cudaStream
     */
    void MoveToCUDAImageAsync(CUDAImage& dst, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param dst
     */
    void CopyToCUDAImage(CUDAImage& dst) const;

    /**
     * @brief
     *
     * @param dst
     * @param cudaStream
     */
    void CopyToCUDAImageAsync(CUDAImage& dst, void* cudaStream = nullptr) const;

    /**
     * @brief
     *
     * @param src
     */
    void MoveFromCUDAImage(CUDAImage& src);

    /**
     * @brief
     *
     * @param src
     * @param cudaStream
     */
    void MoveFromCUDAImageAsync(CUDAImage& src, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param src
     */
    void CopyFromCUDAImage(const CUDAImage& src);

    /**
     * @brief
     *
     * @param src
     * @param cudaStream
     */
    void CopyFromCUDAImageAsync(const CUDAImage& src, void* cudaStream = nullptr);

    /// Cv GPU Mat

    /**
     * @brief
     *
     * @param dst
     */
    void MoveToGpuMat(cv::cuda::GpuMat& dst);

    /**
     * @brief
     *
     * @param dst
     * @param cudaStream
     */
    void MoveToGpuMatAsync(cv::cuda::GpuMat& dst, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param dst
     */
    void CopyToGpuMat(cv::cuda::GpuMat& dst);

    /**
     * @brief
     *
     * @param dst
     * @param cudaStream
     */
    void CopyToGpuMatAsync(cv::cuda::GpuMat& dst, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param src
     */
    void MoveFromGpuMat(cv::cuda::GpuMat& src);

    /**
     * @brief
     *
     * @param src
     * @param cudaStream
     */
    void MoveFromGpuMatAsync(cv::cuda::GpuMat& src, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param src
     */
    void CopyFromGpuMat(const cv::cuda::GpuMat& src);

    /**
     * @brief
     *
     * @param src
     * @param cudaStream
     */
    void CopyFromGpuMatAsync(const cv::cuda::GpuMat& src, void* cudaStream = nullptr);

    /// Cv Mat

    /**
     * @brief
     *
     * @param dst
     */
    void MoveToCvMat(cv::Mat& dst);

    /**
     * @brief
     *
     * @param dst
     * @param cudaStream
     */
    void MoveToCvMatAsync(cv::Mat& dst, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param dst
     */
    void CopyToCvMat(cv::Mat& dst) const;

    /**
     * @brief
     *
     * @param dst
     * @param cudaStream
     */
    void CopyToCvMatAsync(cv::Mat& dst, void* cudaStream = nullptr) const;

    /**
     * @brief
     *
     * @param src
     */
    void MoveFromCvMat(cv::Mat& src);

    /**
     * @brief
     *
     * @param src
     * @param cudaStream
     */
    void MoveFromCvMatAsync(cv::Mat& src, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param src
     */
    void CopyFromCvMat(const cv::Mat& src);

    /**
     * @brief
     *
     * @param src
     * @param cudaStream
     */
    void CopyFromCvMatAsync(const cv::Mat& src, void* cudaStream = nullptr);

    /// Raw host pointer

    /**
     * @brief
     *
     * @param src
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param pitchedAllocation
     */
    void CopyFromRawHostPointer(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation);

    /**
     * @brief
     *
     * @param src
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param pitchedAllocation
     * @param cudaStream
     */
    void CopyFromRawHostPointerAsync(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param dst
     * @param width
     * @param height
     * @param channels
     * @param type
     */
    void CopyToRawHostPointer(void* dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type) const;

    /**
     * @brief
     *
     * @param dst
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param cudaStream
     */
    void CopyToRawHostPointerAsync(void* dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, void* cudaStream = nullptr) const;

    /// Device pointer

    /**
     * @brief
     *
     * @param src
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param pitchedAllocation
     */
    void CopyFromRawDevicePointer(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation);

    /**
     * @brief
     *
     * @param src
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param pitchedAllocation
     * @param cudaStream
     */
    void CopyFromRawDevicePointerAsync(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @param dst
     * @param width
     * @param height
     * @param channels
     * @param type
     */
    void CopyToRawDevicePointer(void* dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type);

    /**
     * @brief
     *
     * @param dst
     * @param width
     * @param height
     * @param channels
     * @param type
     * @param cudaStream
     */
    void CopyToRawDevicePointerAsync(void* dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, void* cudaStream = nullptr);

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] bool Empty() const;

    ///
    size_t width_ = 0;

    ///
    size_t height_ = 0;

    ///
    size_t pitch_ = 0;

    ///
    size_t channels_ = 0;

    ///
    size_t allocatedBytes_ = 0;

    ///
    ELEMENT_TYPE elementType_ = ELEMENT_TYPE::TYPE_UNKNOWN;

    ///
    unsigned char* gpuData_ = nullptr;

    ///
    bool pitchedAllocation_ = false;
};

}

#endif // CUDA_IMAGE_H
