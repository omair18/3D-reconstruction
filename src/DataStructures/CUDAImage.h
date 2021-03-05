/**--------------------------------------------------------------------------------------------------
 * @file	CUDAImage.h.
 *
 * Declares class for working with images on CUDA.
 *-----------------------------------------------------------------------------------------------**/


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

namespace DataStructures
{

struct CUDAImage final
{
    CUDAImage(const CUDAImage& other);

    CUDAImage(CUDAImage&& other) noexcept;

    ~CUDAImage();

    CUDAImage& operator=(const CUDAImage& other);

    CUDAImage& operator=(CUDAImage&& other);

    CUDAImage& operator=(const cv::cuda::GpuMat& other);

    CUDAImage& operator=(cv::cuda::GpuMat&& other);

    CUDAImage& operator=(const cv::Mat& other);

    CUDAImage& operator=(cv::Mat&& other) noexcept;

    bool operator==(const CUDAImage& other);

    bool operator==(const cv::cuda::GpuMat& other);

    void Allocate(unsigned int width, unsigned int height, unsigned int channels, unsigned int elementSize, bool pitchedAllocation);

    void AllocateAsync(unsigned int width, unsigned int height, unsigned int channels, unsigned int elementSize, bool pitchedAllocation, void* cudaStream = nullptr);

    void Release();

    void ReleaseAsync(void* cudaStream = nullptr);

    void MoveToCUDAImage(CUDAImage& dst);

    void MoveToCUDAImageAsync(CUDAImage& dst, void* cudaStream = nullptr);

    void CopyToCUDAImage(CUDAImage& dst);

    void CopyToCUDAImageAsync(CUDAImage& dst, void* cudaStream = nullptr);

    void MoveFromCUDAImage(CUDAImage& src);

    void MoveFromCUDAImageAsync(CUDAImage& src, void* cudaStream = nullptr);

    void CopyFromCUDAImage(const CUDAImage& src);

    void CopyFromCUDAImageAsync(const CUDAImage& src, void* cudaStream = nullptr);

    void MoveFromGpuMat(cv::cuda::GpuMat& src);

    void MoveFromGpuMatAsync(cv::cuda::GpuMat& src, void* cudaStream = nullptr);

    void CopyFromGpuMat(const cv::cuda::GpuMat& src);

    void CopyFromGpuMatAsync(const cv::cuda::GpuMat& src, void* cudaStream = nullptr);

    void MoveFromCvMat(cv::Mat& src);

    void MoveFromCvMatAsync(cv::Mat& src, void* cudaStream = nullptr);

    void CopyFromCvMat(const cv::Mat& src);

    void CopyFromCvMatAsync(const cv::Mat& src, void* cudaStream = nullptr);

    void CopyFromRawPointer(void* src, size_t width, size_t height, size_t channels, size_t elementSize, bool pitchedAllocation);

    void CopyFromRawPointerAsync(void* src, size_t width, size_t height, size_t channels, size_t elementSize, bool pitchedAllocation, void* cudaStream = nullptr);

    void CopyToRawPointer(void* dst, size_t width, size_t height, size_t channels, size_t elementSize);

    void CopyToRawPointerAsync(void* dst, size_t width, size_t height, size_t channels, size_t elementSize, void* cudaStream = nullptr);

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
    size_t elementSize_ = 0;

    ///
    unsigned char* gpuData_ = nullptr;

    ///
    bool pitchedAllocation_ = false;
};

}

#endif // CUDA_IMAGE_H