#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

#include "CUDAImage.h"
#include "Logger.h"

void DataStructures::CUDAImage::Allocate(unsigned int width, unsigned int height, unsigned int channels, unsigned int elementSize, bool pitchedAllocation)
{
    if(gpuData_)
    {
        Release();
    }
    if(pitchedAllocation)
    {
        cudaMallocPitch(&gpuData_, &pitch_, width * channels * elementSize, height);
    }
    else
    {
        cudaMalloc(&gpuData_, width * height * channels * elementSize);
        pitch_ = width * channels * elementSize;
    }
    width_ = width;
    height_  = height;
    channels_ = channels;
    elementSize_ = elementSize;
    pitchedAllocation_ = pitchedAllocation;
}

void DataStructures::CUDAImage::AllocateAsync(unsigned int width, unsigned int height, unsigned int channels, unsigned int elementSize, bool pitchedAllocation, void* cudaStream)
{
    if(gpuData_)
    {
        ReleaseAsync(cudaStream);
    }
    if(pitchedAllocation)
    {
        cudaMallocPitch(&gpuData_, &pitch_, width * channels * elementSize, height);
    }
    else
    {
        cudaMallocAsync(&gpuData_, width * height * channels * elementSize, (cudaStream_t)cudaStream);
        pitch_ = width * channels * elementSize;
    }
    width_ = width;
    height_  = height;
    channels_ = channels;
    elementSize_ = elementSize;
    pitchedAllocation_ = pitchedAllocation;
}

void DataStructures::CUDAImage::Release()
{
    cudaFree(gpuData_);
    gpuData_ = nullptr;
    width_ = 0;
    height_ = 0;
    channels_ = 0;
    pitch_ = 0;
    elementSize_ = 0;
    pitchedAllocation_ = false;
}

void DataStructures::CUDAImage::ReleaseAsync(void* cudaStream)
{
    cudaFreeAsync(gpuData_, (cudaStream_t)cudaStream);
    gpuData_ = nullptr;
    width_ = 0;
    height_ = 0;
    channels_ = 0;
    pitch_ = 0;
    elementSize_ = 0;
    pitchedAllocation_ = false;
}

DataStructures::CUDAImage::CUDAImage(const DataStructures::CUDAImage &other)
{
    Allocate(other.width_, other.height_, other.channels_, other.elementSize_, other.pitchedAllocation_);
    cudaMemcpy2D(gpuData_, pitch_, other.gpuData_, other.pitch_, other.width_ * other.channels_ * other.elementSize_,
                 other.height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

DataStructures::CUDAImage::CUDAImage(DataStructures::CUDAImage &&other) noexcept
{
    std::swap(other.width_, width_);
    other.width_ = 0;

    std::swap(other.height_, height_);
    other.height_ = 0;

    std::swap(other.pitch_, pitch_);
    other.pitch_ = 0;

    std::swap(other.channels_, channels_);
    other.channels_ = 0;

    std::swap(other.elementSize_, elementSize_);
    other.elementSize_ = 0;

    std::swap(other.gpuData_, gpuData_);
    other.gpuData_ = nullptr;

    std::swap(other.pitchedAllocation_, pitchedAllocation_);
    other.pitchedAllocation_ = false;
}

DataStructures::CUDAImage::~CUDAImage()
{
    if(gpuData_)
    {
        Release();
    }
}

DataStructures::CUDAImage &DataStructures::CUDAImage::operator=(const DataStructures::CUDAImage &other)
{
    if (this == &other)
    {
        return *this;
    }
    CopyFromCUDAImage(other);
    return *this;
}

DataStructures::CUDAImage &DataStructures::CUDAImage::operator=(DataStructures::CUDAImage &&other)
{
    if (gpuData_ == other.gpuData_)
    {
        return *this;
    }
    MoveFromCUDAImage(other);
    return *this;
}

DataStructures::CUDAImage &DataStructures::CUDAImage::operator=(const cv::cuda::GpuMat &other)
{
    if (this == (CUDAImage*)(&other))
    {
        return *this;
    }
    CopyFromGpuMat(other);
    return *this;
}

DataStructures::CUDAImage &DataStructures::CUDAImage::operator=(cv::cuda::GpuMat &&other)
{
    if (gpuData_ == other.data)
    {
        return *this;
    }
    MoveFromGpuMat(other);
    return *this;
}

DataStructures::CUDAImage &DataStructures::CUDAImage::operator=(const cv::Mat &other)
{
    if (&other == (cv::Mat*)this)
        return *this;
    CopyFromCvMat(other);
    return *this;
}

DataStructures::CUDAImage &DataStructures::CUDAImage::operator=(cv::Mat &&other) noexcept
{
    MoveFromCvMat(other);
    return *this;
}

bool DataStructures::CUDAImage::operator==(const DataStructures::CUDAImage &other)
{
    return width_ == other.width_ &&
         height_ == other.height_ &&
         pitch_ == other.pitch_ &&
         channels_ == other.channels_ &&
         elementSize_ == other.elementSize_;
}

bool DataStructures::CUDAImage::operator==(const cv::cuda::GpuMat& other)
{
    return width_ == other.cols &&
           height_ == other.rows &&
           pitch_ == other.step &&
           channels_ == other.channels() &&
           elementSize_ == other.elemSize();
}

void DataStructures::CUDAImage::MoveFromGpuMat(cv::cuda::GpuMat &src)
{
    if(gpuData_ == src.data)
    {
        return;
    }

    if(gpuData_)
    {
        Release();
    }

    std::swap(gpuData_, src.data);
    src.data = nullptr;

    width_ = src.cols;
    src.cols = 0;

    height_ = src.rows;
    src.rows = 0;

    pitch_ = src.step;
    src.step = 0;

    elementSize_ = src.elemSize();

    pitchedAllocation_ = width_ * channels_ * elementSize_ != pitch_;
}

void DataStructures::CUDAImage::CopyFromGpuMat(const cv::cuda::GpuMat &src)
{
    bool pitchedAllocation = src.cols * src.channels() * src.elemSize() != src.step;

    if(gpuData_)
    {
        bool needReallocation =
                width_ != src.cols ||
                height_ != src.rows ||
                pitch_ != src.step ||
                channels_ != src.channels() ||
                elementSize_ != src.elemSize();
        if(needReallocation)
        {
            Allocate(src.cols, src.rows, src.channels(), src.elemSize(), pitchedAllocation);
        }
    }
    else
    {
        Allocate(src.cols, src.rows, src.channels(), src.elemSize(), pitchedAllocation);
    }
    cudaMemcpy2D(gpuData_, pitch_, src.data, src.step, src.cols * src.channels() * src.elemSize(),
                 src.rows, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void DataStructures::CUDAImage::MoveFromCvMat(cv::Mat &src)
{
    CopyFromCvMat(src);
    src.release();
}

void DataStructures::CUDAImage::CopyFromCvMat(const cv::Mat &src)
{
    if(gpuData_)
    {
        bool needReallocation =
                width_ != src.cols ||
                height_ != src.rows ||
                channels_ != src.channels() ||
                elementSize_ != src.elemSize();
        if(needReallocation)
        {
            Allocate(src.cols, src.rows, src.channels(), src.elemSize(), true);
        }
    }
    else
    {
        Allocate(src.cols, src.rows, src.channels(), src.elemSize(), true);
    }
    cudaMemcpy2D(gpuData_, pitch_, src.data, src.cols * src.channels() * src.elemSize(),
                 src.cols * src.channels() * src.elemSize(), src.rows, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

bool DataStructures::CUDAImage::Empty() const
{
    return gpuData_ == nullptr;
}

void DataStructures::CUDAImage::CopyFromRawPointer(void* src, size_t width, size_t height, size_t channels, size_t elementSize, bool pitchedAllocation)
{
    if(gpuData_)
    {
        bool needReallocation =
                width_ != width ||
                height_ != height ||
                channels_ != channels ||
                elementSize_ != elementSize;
        needReallocation |= (pitchedAllocation != (width_ * channels_ * elementSize_ != pitch_));
        if(needReallocation)
        {
            Allocate(width, height, channels, elementSize, pitchedAllocation);
        }
    }
    else
    {
        Allocate(width, height, channels, elementSize, pitchedAllocation);
    }
    cudaMemcpy2D(gpuData_, pitch_, src, width * channels * elementSize, width * channels * elementSize, height,
                 cudaMemcpyKind::cudaMemcpyHostToDevice);

}

void DataStructures::CUDAImage::MoveFromGpuMatAsync(cv::cuda::GpuMat &src, void *cudaStream)
{
    if(gpuData_ == src.data)
    {
        return;
    }

    if(gpuData_)
    {
        ReleaseAsync(cudaStream);
    }

    std::swap(gpuData_, src.data);
    src.data = nullptr;

    width_ = src.cols;
    src.cols = 0;

    height_ = src.rows;
    src.rows = 0;

    pitch_ = src.step;
    src.step = 0;

    elementSize_ = src.elemSize();

    pitchedAllocation_ = width_ * channels_ * elementSize_ != pitch_;
}

void DataStructures::CUDAImage::CopyFromGpuMatAsync(const cv::cuda::GpuMat &src, void *cudaStream)
{
    bool pitchedAllocation = src.cols * src.channels() * src.elemSize() != src.step;

    if(gpuData_)
    {
        bool needReallocation =
                width_ != src.cols ||
                height_ != src.rows ||
                pitch_ != src.step ||
                channels_ != src.channels() ||
                elementSize_ != src.elemSize();
        if(needReallocation)
        {
            AllocateAsync(src.cols, src.rows, src.channels(), src.elemSize(), pitchedAllocation, cudaStream);
        }
    }
    else
    {
        AllocateAsync(src.cols, src.rows, src.channels(), src.elemSize(), pitchedAllocation, cudaStream);
    }
    cudaMemcpy2DAsync(gpuData_, pitch_, src.data, src.step, src.cols * src.channels() * src.elemSize(),
                 src.rows, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
}

void DataStructures::CUDAImage::MoveFromCvMatAsync(cv::Mat &src, void *cudaStream)
{
    CopyFromCvMatAsync(src, cudaStream);
    src.release();
}

void DataStructures::CUDAImage::CopyFromCvMatAsync(const cv::Mat &src, void *cudaStream)
{
    if(gpuData_)
    {
        bool needReallocation =
                width_ != src.cols ||
                height_ != src.rows ||
                channels_ != src.channels() ||
                elementSize_ != src.elemSize();
        if(needReallocation)
        {
            AllocateAsync(src.cols, src.rows, src.channels(), src.elemSize(), true, cudaStream);
        }
    }
    else
    {
        AllocateAsync(src.cols, src.rows, src.channels(), src.elemSize(), true, cudaStream);
    }
    cudaMemcpy2DAsync(gpuData_, pitch_, src.data, src.cols * src.channels() * src.elemSize(),
                 src.cols * src.channels() * src.elemSize(), src.rows, cudaMemcpyKind::cudaMemcpyHostToDevice, (cudaStream_t)cudaStream);
}

void DataStructures::CUDAImage::CopyFromRawPointerAsync(void *src, size_t width, size_t height, size_t channels, size_t elementSize, bool pitchedAllocation, void *cudaStream)
{
    if(gpuData_)
    {
        bool needReallocation =
                width_ != width ||
                height_ != height ||
                channels_ != channels ||
                elementSize_ != elementSize;
        needReallocation |= (pitchedAllocation != (width_ * channels_ * elementSize_ != pitch_));
        if(needReallocation)
        {
            AllocateAsync(width, height, channels, elementSize, pitchedAllocation, cudaStream);
        }
    }
    else
    {
        AllocateAsync(width, height, channels, elementSize, pitchedAllocation, cudaStream);
    }
    cudaMemcpy2DAsync(gpuData_, pitch_, src, width * channels * elementSize, width * channels * elementSize, height,
                 cudaMemcpyKind::cudaMemcpyHostToDevice, (cudaStream_t)cudaStream);
}

void DataStructures::CUDAImage::CopyToRawPointer(void *dst, size_t width, size_t height, size_t channels, size_t elementSize)
{
    if(gpuData_)
    {
        cudaMemcpy2D(dst, width * channels * elementSize, gpuData_, pitch_, width_ * channels_ * elementSize_,
                     height_, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }
}

void DataStructures::CUDAImage::CopyToRawPointerAsync(void *dst, size_t width, size_t height, size_t channels, size_t elementSize, void *cudaStream)
{
    if(gpuData_)
    {
        cudaMemcpy2DAsync(dst, width * channels * elementSize, gpuData_, pitch_, width_ * channels_ * elementSize_,
                     height_, cudaMemcpyKind::cudaMemcpyDeviceToHost, (cudaStream_t)cudaStream);
    }
}

void DataStructures::CUDAImage::MoveToCUDAImage(DataStructures::CUDAImage &dst)
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        dst.Release();
    }

    std::swap(dst.width_, width_);
    width_ = 0;

    std::swap(dst.height_, height_);
    height_ = 0;

    std::swap(dst.pitch_, pitch_);
    pitch_ = 0;

    std::swap(dst.channels_, channels_);
    channels_ = 0;

    std::swap(dst.elementSize_, elementSize_);
    elementSize_ = 0;

    std::swap(dst.gpuData_, gpuData_);
    gpuData_ = nullptr;

    std::swap(dst.pitchedAllocation_, pitchedAllocation_);
    pitchedAllocation_ = false;
}

void DataStructures::CUDAImage::MoveToCUDAImageAsync(DataStructures::CUDAImage &dst, void *cudaStream)
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        dst.ReleaseAsync(cudaStream);
    }

    std::swap(dst.width_, width_);
    width_ = 0;

    std::swap(dst.height_, height_);
    height_ = 0;

    std::swap(dst.pitch_, pitch_);
    pitch_ = 0;

    std::swap(dst.channels_, channels_);
    channels_ = 0;

    std::swap(dst.elementSize_, elementSize_);
    elementSize_ = 0;

    std::swap(dst.gpuData_, gpuData_);
    gpuData_ = nullptr;

    std::swap(dst.pitchedAllocation_, pitchedAllocation_);
    pitchedAllocation_ = false;
}

void DataStructures::CUDAImage::CopyToCUDAImage(DataStructures::CUDAImage &dst)
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        bool needReallocation =
                width_ != dst.width_ ||
                height_ != dst.height_ ||
                pitch_ != dst.pitch_ ||
                channels_ != dst.channels_ ||
                elementSize_ != dst.elementSize_;
        if(needReallocation)
        {
            dst.Allocate(width_, height_, channels_, elementSize_, pitchedAllocation_);
        }
    }
    else
    {
        dst.Allocate(width_, height_, channels_, elementSize_, pitchedAllocation_);
    }
    cudaMemcpy2D(dst.gpuData_, dst.pitch_, gpuData_, pitch_, width_ * channels_ * elementSize_,
                      height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void DataStructures::CUDAImage::CopyToCUDAImageAsync(DataStructures::CUDAImage &dst, void *cudaStream)
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        bool needReallocation =
                width_ != dst.width_ ||
                height_ != dst.height_ ||
                pitch_ != dst.pitch_ ||
                channels_ != dst.channels_ ||
                elementSize_ != dst.elementSize_;
        if(needReallocation)
        {
            dst.AllocateAsync(width_, height_, channels_, elementSize_, pitchedAllocation_, cudaStream);
        }
    }
    else
    {
        dst.AllocateAsync(width_, height_, channels_, elementSize_, pitchedAllocation_, cudaStream);
    }
    cudaMemcpy2DAsync(dst.gpuData_, dst.pitch_, gpuData_, pitch_, width_ * channels_ * elementSize_,
                 height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
}

void DataStructures::CUDAImage::MoveFromCUDAImage(DataStructures::CUDAImage &src)
{
    if (gpuData_ == src.gpuData_)
    {
        return;
    }

    if(gpuData_)
    {
        Release();
    }

    std::swap(width_, src.width_);
    src.width_ = 0;

    std::swap(height_, src.height_);
    src.height_ = 0;

    std::swap(pitch_, src.pitch_);
    src.pitch_ = 0;

    std::swap(channels_, src.channels_);
    src.channels_ = 0;

    std::swap(elementSize_, src.elementSize_);
    src.elementSize_ = 0;

    std::swap(gpuData_, src.gpuData_);
    src.gpuData_ = nullptr;

    std::swap(pitchedAllocation_, src.pitchedAllocation_);
    src.pitchedAllocation_ = false;

}

void DataStructures::CUDAImage::MoveFromCUDAImageAsync(DataStructures::CUDAImage &src, void *cudaStream)
{
    if (gpuData_ == src.gpuData_)
    {
        return;
    }

    if(gpuData_)
    {
        ReleaseAsync(cudaStream);
    }

    std::swap(width_, src.width_);
    src.width_ = 0;

    std::swap(height_, src.height_);
    src.height_ = 0;

    std::swap(pitch_, src.pitch_);
    src.pitch_ = 0;

    std::swap(channels_, src.channels_);
    src.channels_ = 0;

    std::swap(elementSize_, src.elementSize_);
    src.elementSize_ = 0;

    std::swap(gpuData_, src.gpuData_);
    src.gpuData_ = nullptr;

    std::swap(pitchedAllocation_, src.pitchedAllocation_);
    src.pitchedAllocation_ = false;
}

void DataStructures::CUDAImage::CopyFromCUDAImage(const DataStructures::CUDAImage& src)
{
    if(gpuData_ == src.gpuData_)
    {
        return;
    }
    if(gpuData_)
    {
        bool needReallocation =
                width_ != src.width_ ||
                height_ != src.height_ ||
                pitch_ != src.pitch_ ||
                channels_ != src.channels_ ||
                elementSize_ != src.elementSize_;
        if(needReallocation)
        {
            Allocate(src.width_, src.height_, src.channels_, src.elementSize_, src.pitchedAllocation_);
        }
    }
    else
    {
        Allocate(src.width_, src.height_, src.channels_, src.elementSize_, src.pitchedAllocation_);
    }
    cudaMemcpy2D(gpuData_, pitch_, src.gpuData_, src.pitch_, src.width_ * src.channels_ * src.elementSize_,
                 src.height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void DataStructures::CUDAImage::CopyFromCUDAImageAsync(const DataStructures::CUDAImage& src, void* cudaStream)
{
    if(gpuData_ == src.gpuData_)
    {
        return;
    }

    if(gpuData_)
    {
        bool needReallocation =
                width_ != src.width_ ||
                height_ != src.height_ ||
                pitch_ != src.pitch_ ||
                channels_ != src.channels_ ||
                elementSize_ != src.elementSize_;
        if(needReallocation)
        {
            AllocateAsync(src.width_, src.height_, src.channels_, src.elementSize_, src.pitchedAllocation_, cudaStream);
        }
    }
    else
    {
        AllocateAsync(src.width_, src.height_, src.channels_, src.elementSize_, src.pitchedAllocation_, cudaStream);
    }
    cudaMemcpy2DAsync(gpuData_, pitch_, src.gpuData_, src.pitch_, src.width_ * src.channels_ * src.elementSize_,
                 src.height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
}


