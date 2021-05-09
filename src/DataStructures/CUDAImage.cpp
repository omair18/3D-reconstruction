#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

#include "CUDAImage.h"
#include "Logger.h"

namespace DataStructures
{

static size_t GetTypeSize(CUDAImage::ELEMENT_TYPE type)
{
    switch (type)
    {
        case CUDAImage::ELEMENT_TYPE::TYPE_8S:
        case CUDAImage::ELEMENT_TYPE::TYPE_8U:
            return 1;

        case CUDAImage::ELEMENT_TYPE::TYPE_16S:
        case CUDAImage::ELEMENT_TYPE::TYPE_16U:
        case CUDAImage::ELEMENT_TYPE::TYPE_16F:
            return 2;

        case CUDAImage::ELEMENT_TYPE::TYPE_32F:
        case CUDAImage::ELEMENT_TYPE::TYPE_32S:
            return 4;

        case CUDAImage::ELEMENT_TYPE::TYPE_64F:
            return 8;

        case CUDAImage::ELEMENT_TYPE::TYPE_UNKNOWN:
        default:
            return 0;
    }
}

static std::pair<CUDAImage::ELEMENT_TYPE, size_t> ConvertCvTypeToCUDAImageElementTypeAndChannels(int type)
{
    switch (type)
    {
        case CV_8UC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8U, 1);
        case CV_8SC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8S, 1);
        case CV_8UC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8U, 2);
        case CV_8SC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8S, 2);
        case CV_8UC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8U, 3);
        case CV_8SC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8S, 3);
        case CV_8UC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8U, 4);
        case CV_8SC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_8S, 4);
        case CV_16FC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16F, 1);
        case CV_16UC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16U, 1);
        case CV_16SC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16S, 1);
        case CV_16FC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16F, 2);
        case CV_16UC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16U, 2);
        case CV_16SC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16S, 2);
        case CV_16FC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16F, 3);
        case CV_16UC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16U, 3);
        case CV_16SC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16S, 3);
        case CV_16FC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16F, 4);
        case CV_16UC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16U, 4);
        case CV_16SC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_16S, 4);
        case CV_32FC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32F, 1);
        case CV_32SC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32S, 1);
        case CV_32FC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32F, 2);
        case CV_32SC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32S, 2);
        case CV_32FC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32F, 3);
        case CV_32SC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32S, 3);
        case CV_32FC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32F, 4);
        case CV_32SC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_32S, 4);
        case CV_64FC1:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_64F, 1);
        case CV_64FC2:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_64F, 2);
        case CV_64FC3:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_64F, 3);
        case CV_64FC4:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_64F, 4);
        default:
            return std::make_pair(CUDAImage::ELEMENT_TYPE::TYPE_UNKNOWN, 0);
    }
}

static int ConvertCUDAImageElementTypeAndChannelsToCvType(CUDAImage::ELEMENT_TYPE type, size_t channels)
{
    if(type == CUDAImage::ELEMENT_TYPE::TYPE_8U)
    {
        if(channels == 1)
            return CV_8UC1;
        if(channels == 2)
            return CV_8UC2;
        if(channels == 3)
            return CV_8UC3;
        if(channels == 4)
            return CV_8UC4;
    }

    if(type == CUDAImage::ELEMENT_TYPE::TYPE_8S)
    {
        if(channels == 1)
            return CV_8SC1;
        if(channels == 2)
            return CV_8SC2;
        if(channels == 3)
            return CV_8SC3;
        if(channels == 4)
            return CV_8SC4;
    }

    if(type == CUDAImage::ELEMENT_TYPE::TYPE_16S)
    {
        if(channels == 1)
            return CV_16SC1;
        if(channels == 2)
            return CV_16SC2;
        if(channels == 3)
            return CV_16SC3;
        if(channels == 4)
            return CV_16SC4;
    }

    if(type == CUDAImage::ELEMENT_TYPE::TYPE_16U)
    {
        if(channels == 1)
            return CV_16UC1;
        if(channels == 2)
            return CV_16UC2;
        if(channels == 3)
            return CV_16UC3;
        if(channels == 4)
            return CV_16UC4;
    }

    if(type == CUDAImage::ELEMENT_TYPE::TYPE_16F)
    {
        if(channels == 1)
            return CV_16FC1;
        if(channels == 2)
            return CV_16FC2;
        if(channels == 3)
            return CV_16FC3;
        if(channels == 4)
            return CV_16FC4;
    }

    if(type == CUDAImage::ELEMENT_TYPE::TYPE_32S)
    {
        if(channels == 1)
            return CV_32SC1;
        if(channels == 2)
            return CV_32SC2;
        if(channels == 3)
            return CV_32SC3;
        if(channels == 4)
            return CV_32SC4;
    }

    if(type == CUDAImage::ELEMENT_TYPE::TYPE_32F)
    {
        if(channels == 1)
            return CV_32FC1;
        if(channels == 2)
            return CV_32FC2;
        if(channels == 3)
            return CV_32FC3;
        if(channels == 4)
            return CV_32FC4;
    }

    if(type == CUDAImage::ELEMENT_TYPE::TYPE_64F)
    {
        if(channels == 1)
            return CV_64FC1;
        if(channels == 2)
            return CV_64FC2;
        if(channels == 3)
            return CV_64FC3;
        if(channels == 4)
            return CV_64FC4;
    }

    return -1;
}

static double BytesToMB(size_t bytes)
{
    return static_cast<double>(bytes) / 1024. / 1024.;
}

CUDAImage::CUDAImage() :
width_(0),
height_(0),
pitch_(0),
channels_(0),
allocatedBytes_(0),
elementType_(TYPE_UNKNOWN),
gpuData_(nullptr),
pitchedAllocation_(false)
{

}

void CUDAImage::Allocate(unsigned int width, unsigned int height, unsigned int channels, ELEMENT_TYPE type, bool pitchedAllocation)
{
    cudaError_t status;
    if(gpuData_)
    {
        Release();
    }
    if(pitchedAllocation)
    {
        status = cudaMallocPitch(&gpuData_, &pitch_, width * channels * GetTypeSize(type), height);
        if(status == cudaError_t::cudaSuccess)
        {
            LOG_TRACE() << "Successfully allocated pitched memory for " << width << "x" << height << " image."
            << " Size: " << BytesToMB(height * pitch_) << "MB; Pitch: " << pitch_ << ".";
        }
        else
        {
            LOG_ERROR() << "Failed to allocate pitched memory for " << width << "x" << height <<" image."
                        << "CUDA error " << static_cast<int>(status) << " - " << cudaGetErrorName(status)
                        << ": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    else
    {
        status = cudaMalloc(&gpuData_, width * height * channels * GetTypeSize(type));
        if(status == cudaError_t::cudaSuccess)
        {
            LOG_TRACE() << "Successfully allocated linear memory for " << width << "x" << height << " image."
                        << " Size: " << BytesToMB(height * width * channels * GetTypeSize(type)) << "MB.";
        }
        else
        {
            LOG_ERROR() << "Failed to allocate linear memory for " << width << "x" << height <<" image."
                        << "CUDA error " << static_cast<int>(status) << " - " << cudaGetErrorName(status)
                        << ": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
        pitch_ = width * channels * GetTypeSize(type);
    }
    width_ = width;
    height_  = height;
    channels_ = channels;
    elementType_ = type;
    pitchedAllocation_ = pitchedAllocation;
    allocatedBytes_ = pitch_ * height_;
}

void CUDAImage::AllocateAsync(unsigned int width, unsigned int height, unsigned int channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream)
{
    cudaError_t status;
    if(gpuData_)
    {
        ReleaseAsync(cudaStream);
    }
    if(pitchedAllocation)
    {
        status = cudaMallocPitch(&gpuData_, &pitch_, width * channels * GetTypeSize(type), height);
        if(status == cudaError_t::cudaSuccess)
        {
            LOG_TRACE() << "Successfully allocated pitched memory for " << width << "x" << height << " image."
                        << " Size: " << BytesToMB(height * pitch_) << "MB; Pitch: " << pitch_ << ".";
        }
        else
        {
            LOG_ERROR() << "Failed to allocate pitched memory for " << width << "x" << height <<" image."
                        << "CUDA error " << static_cast<int>(status) << " - " << cudaGetErrorName(status)
                        << ": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    else
    {
        status = cudaMallocAsync(&gpuData_, width * height * channels * GetTypeSize(type), (cudaStream_t)cudaStream);
        if(status == cudaError_t::cudaSuccess)
        {
            LOG_TRACE() << "Successfully allocated linear memory for " << width << "x" << height << " image asynchronously. Size: "
            << BytesToMB(height * width * channels * GetTypeSize(type)) << "MB.";
        }
        else
        {
            throw std::runtime_error("CUDA error.");
        }
        pitch_ = width * channels * GetTypeSize(type);
    }
    width_ = width;
    height_  = height;
    channels_ = channels;
    elementType_ = type;
    pitchedAllocation_ = pitchedAllocation;
    allocatedBytes_ = pitch_ * height_;
}

void CUDAImage::Release()
{
    auto status = cudaFree(gpuData_);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to release " << BytesToMB(allocatedBytes_) << " MB of GPU memory. "
        << "CUDA error " << static_cast<int>(status) << " - " << cudaGetErrorName(status)
        << ": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    else
    {
        LOG_TRACE() << "Successfully released " << BytesToMB(allocatedBytes_) << " MB of GPU memory." ;
    }
    gpuData_ = nullptr;
    width_ = 0;
    height_ = 0;
    channels_ = 0;
    pitch_ = 0;
    elementType_ = TYPE_UNKNOWN;
    pitchedAllocation_ = false;
    allocatedBytes_ = 0;
}

void CUDAImage::ReleaseAsync(void* cudaStream)
{
    auto status = cudaFreeAsync(gpuData_, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to release " << BytesToMB(allocatedBytes_) << " MB of GPU memory asynchronously. "
                    << "CUDA error " << static_cast<int>(status) << " - " << cudaGetErrorName(status)
                    << ": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    else
    {
        LOG_TRACE() << "Successfully released " << BytesToMB(allocatedBytes_) << " MB of GPU memory asynchronously.";
    }
    gpuData_ = nullptr;
    width_ = 0;
    height_ = 0;
    channels_ = 0;
    pitch_ = 0;
    elementType_ = TYPE_UNKNOWN;
    pitchedAllocation_ = false;
    allocatedBytes_ = 0;
}

CUDAImage::CUDAImage(const CUDAImage& other)
{
    Allocate(other.width_, other.height_, other.channels_, other.elementType_, other.pitchedAllocation_);
    auto status = cudaMemcpy2D(gpuData_, pitch_, other.gpuData_, other.pitch_, other.width_ * other.channels_ * GetTypeSize(other.elementType_),
                 other.height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if (status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to make copy of image. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
}

CUDAImage::CUDAImage(CUDAImage&& other) noexcept
{
    width_ = other.width_;
    height_ = other.height_;
    pitch_ = other.pitch_;
    channels_ = other.channels_;
    elementType_ = other.elementType_;
    gpuData_ = other.gpuData_;
    allocatedBytes_ = other.allocatedBytes_;
    pitchedAllocation_ = other.pitchedAllocation_;

    other.width_ = 0;
    other.height_ = 0;
    other.pitch_ = 0;
    other.channels_ = 0;
    other.elementType_ = TYPE_UNKNOWN;
    other.gpuData_ = nullptr;
    other.allocatedBytes_ = 0;
    other.pitchedAllocation_ = false;
}

CUDAImage::~CUDAImage()
{
    if(gpuData_)
    {
        Release();
    }
}

CUDAImage& CUDAImage::operator=(const CUDAImage& other)
{
    if (this == &other)
    {
        return *this;
    }
    CopyFromCUDAImage(other);
    return *this;
}

CUDAImage& CUDAImage::operator=(CUDAImage&& other) noexcept(false)
{
    if (gpuData_ == other.gpuData_)
    {
        return *this;
    }
    MoveFromCUDAImage(other);
    return *this;
}

CUDAImage& CUDAImage::operator=(const cv::cuda::GpuMat& other)
{
    if (this == reinterpret_cast<const CUDAImage*>(&other))
    {
        return *this;
    }
    CopyFromGpuMat(other);
    return *this;
}

CUDAImage& CUDAImage::operator=(cv::cuda::GpuMat&& other)
{
    if (gpuData_ == other.data)
    {
        return *this;
    }
    MoveFromGpuMat(other);
    return *this;
}

CUDAImage& CUDAImage::operator=(const cv::Mat& other)
{
    if (&other == (cv::Mat*)this)
    {
        return *this;
    }
    CopyFromCvMat(other);
    return *this;
}

CUDAImage& CUDAImage::operator=(cv::Mat&& other) noexcept
{
    MoveFromCvMat(other);
    return *this;
}

void CUDAImage::MoveFromGpuMat(cv::cuda::GpuMat& src)
{
    if(gpuData_ == src.data)
    {
        return;
    }

    if(gpuData_)
    {
        Release();
    }

    auto [type, channels] = ConvertCvTypeToCUDAImageElementTypeAndChannels(src.type());
    gpuData_ = src.data;
    width_ = src.cols;
    height_ = src.rows;
    pitch_ = src.step;
    elementType_ = type;
    channels_ = channels;
    pitchedAllocation_ = width_ * channels_ * GetTypeSize(elementType_) != pitch_;
    allocatedBytes_ = pitch_ * height_;

    src.data = nullptr;
    src.datastart = nullptr;
    src.dataend = nullptr;
    src.cols = 0;
    src.rows = 0;
    src.step = 0;
    src.flags = 0;
}

void CUDAImage::CopyFromGpuMat(const cv::cuda::GpuMat& src)
{
    bool pitchedAllocation = src.cols * src.channels() * src.elemSize1() != src.step;
    auto [type, channels] = ConvertCvTypeToCUDAImageElementTypeAndChannels(src.type());

    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < src.rows * src.cols * src.channels() * src.elemSize1();
        if(needReallocation)
        {
            Allocate(src.cols, src.rows, src.channels(), type, pitchedAllocation);
        }
    }
    else
    {
        Allocate(src.cols, src.rows, src.channels(), type, pitchedAllocation);
    }
    auto status = cudaMemcpy2D(gpuData_, pitch_, src.data, src.step, src.cols * src.channels() * src.elemSize1(),
                 src.rows, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from cv::cuda::GpuMat. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = src.cols;
    height_ = src.rows;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::MoveFromCvMat(cv::Mat& src)
{
    CopyFromCvMat(src);

    src.release();
    src.data = nullptr;
    src.datastart = nullptr;
    src.dataend = nullptr;
    src.cols = 0;
    src.rows = 0;
    src.step = 0;
    src.flags = 0;
}

void CUDAImage::CopyFromCvMat(const cv::Mat& src)
{
    auto [type, channels] = ConvertCvTypeToCUDAImageElementTypeAndChannels(src.type());
    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < src.rows * src.cols * src.channels() * src.elemSize1();
        if(needReallocation)
        {
            Allocate(src.cols, src.rows, src.channels(), type, true);
        }
    }
    else
    {
        Allocate(src.cols, src.rows, src.channels(), type, true);
    }
    auto status = cudaMemcpy2D(gpuData_, pitch_, src.data, src.cols * src.channels() * src.elemSize1(),
                 src.cols * src.channels() * src.elemSize1(), src.rows, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from cv::Mat. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = src.cols;
    height_ = src.rows;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::CopyFromRawHostPointer(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation)
{
    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < width * height * channels * GetTypeSize(type);
        if(needReallocation)
        {
            Allocate(width, height, channels, type, pitchedAllocation);
        }
    }
    else
    {
        Allocate(width, height, channels, type, pitchedAllocation);
    }
    auto status = cudaMemcpy2D(gpuData_, pitch_, src, width * channels * GetTypeSize(type), width * channels * GetTypeSize(type), height,
                 cudaMemcpyKind::cudaMemcpyHostToDevice);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from raw host pointer. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = width;
    height_ = height;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::MoveFromGpuMatAsync(cv::cuda::GpuMat& src, void* cudaStream)
{
    if(gpuData_ == src.data)
    {
        return;
    }

    if(gpuData_)
    {
        ReleaseAsync(cudaStream);
    }

    auto [type, channels] = ConvertCvTypeToCUDAImageElementTypeAndChannels(src.type());

    gpuData_ = src.data;
    width_ = src.cols;
    height_ = src.rows;
    pitch_ = src.step;
    elementType_ = type;
    channels_ = channels;
    pitchedAllocation_ = width_ * channels_ * GetTypeSize(elementType_) != pitch_;
    allocatedBytes_ = pitch_ * height_;

    src.data = nullptr;
    src.datastart = nullptr;
    src.dataend = nullptr;
    src.cols = 0;
    src.rows = 0;
    src.step = 0;
    src.flags = 0;
}

void CUDAImage::CopyFromGpuMatAsync(const cv::cuda::GpuMat& src, void* cudaStream)
{
    bool pitchedAllocation = src.cols * src.channels() * src.elemSize1() != src.step;
    auto [type, channels] = ConvertCvTypeToCUDAImageElementTypeAndChannels(src.type());

    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < src.rows * src.cols * src.channels() * src.elemSize1();
        if(needReallocation)
        {
            AllocateAsync(src.cols, src.rows, src.channels(), type, pitchedAllocation, cudaStream);
        }
    }
    else
    {
        AllocateAsync(src.cols, src.rows, src.channels(), type, pitchedAllocation, cudaStream);
    }
    auto status = cudaMemcpy2DAsync(gpuData_, pitch_, src.data, src.step, src.cols * src.channels() * src.elemSize1(),
                 src.rows, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from cv::cuda::GpuMat asynchronously. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = src.cols;
    height_ = src.rows;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::MoveFromCvMatAsync(cv::Mat& src, void* cudaStream)
{
    CopyFromCvMatAsync(src, cudaStream);
    src.data = nullptr;
    src.datastart = nullptr;
    src.dataend = nullptr;

    src.rows = 0;
    src.cols = 0;
    src.step = 0;
    src.dims = 0;
    src.flags = 0;
}

void CUDAImage::CopyFromCvMatAsync(const cv::Mat& src, void* cudaStream)
{
    auto [type, channels] = ConvertCvTypeToCUDAImageElementTypeAndChannels(src.type());
    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < src.rows * src.cols * src.channels() * src.elemSize1();
        if(needReallocation)
        {
            AllocateAsync(src.cols, src.rows, src.channels(), type, true, cudaStream);
        }
    }
    else
    {
        AllocateAsync(src.cols, src.rows, src.channels(), type, true, cudaStream);
    }
    auto status = cudaMemcpy2DAsync(gpuData_, pitch_, src.data, src.cols * src.channels() * src.elemSize1(),
                 src.cols * src.channels() * src.elemSize1(), src.rows, cudaMemcpyKind::cudaMemcpyHostToDevice, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from cv::Mat asynchronously. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = src.cols;
    height_ = src.rows;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::CopyFromRawHostPointerAsync(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream)
{
    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < width * height * channels * GetTypeSize(type);
        if(needReallocation)
        {
            AllocateAsync(width, height, channels, type, pitchedAllocation, cudaStream);
        }
    }
    else
    {
        AllocateAsync(width, height, channels, type, pitchedAllocation, cudaStream);
    }
    auto status = cudaMemcpy2DAsync(gpuData_, pitch_, src, width * channels * GetTypeSize(type), width * channels * GetTypeSize(type), height,
                 cudaMemcpyKind::cudaMemcpyHostToDevice, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from raw host pointer asynchronously. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = width;
    height_ = height;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::CopyToRawHostPointer(void* dst, size_t width, size_t channels, ELEMENT_TYPE type) const
{
    if(gpuData_)
    {
        auto status = cudaMemcpy2D(dst, width * channels * GetTypeSize(type), gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                     height_, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to copy image's data to raw host pointer. CUDA error " << static_cast<int>(status) << " - "
            << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    else
    {
        LOG_WARNING() << "Nothing to copy. Image is empty.";
    }
}

void CUDAImage::CopyToRawHostPointerAsync(void* dst, size_t width, size_t channels, ELEMENT_TYPE type, void* cudaStream) const
{
    if(gpuData_)
    {
        auto status = cudaMemcpy2DAsync(dst, width * channels * GetTypeSize(type), gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                     height_, cudaMemcpyKind::cudaMemcpyDeviceToHost, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to copy image's data to raw host pointer asynchronously. CUDA error " << static_cast<int>(status) << " - "
                        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    else
    {
        LOG_WARNING() << "Nothing to copy. Image is empty.";
    }
}

void CUDAImage::MoveToCUDAImage(CUDAImage& dst)
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        dst.Release();
    }

    dst.width_ = width_;
    dst.height_ = height_;
    dst.pitch_ = pitch_;
    dst.channels_ = channels_;
    dst.elementType_ = elementType_;
    dst.gpuData_ = gpuData_;
    dst.allocatedBytes_ = allocatedBytes_;
    dst.pitchedAllocation_ = pitchedAllocation_;

    width_ = 0;
    height_ = 0;
    pitch_ = 0;
    channels_ = 0;
    elementType_ = TYPE_UNKNOWN;
    gpuData_ = nullptr;
    allocatedBytes_ = 0;
    pitchedAllocation_ = false;
}

void CUDAImage::MoveToCUDAImageAsync(CUDAImage& dst, void* cudaStream)
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        dst.ReleaseAsync(cudaStream);
    }

    dst.width_ = width_;
    dst.height_ = height_;
    dst.pitch_ = pitch_;
    dst.channels_ = channels_;
    dst.elementType_ = elementType_;
    dst.gpuData_ = gpuData_;
    dst.allocatedBytes_ = allocatedBytes_;
    dst.pitchedAllocation_ = pitchedAllocation_;

    width_ = 0;
    height_ = 0;
    pitch_ = 0;
    channels_ = 0;
    elementType_ = TYPE_UNKNOWN;
    gpuData_ = nullptr;
    allocatedBytes_ = 0;
    pitchedAllocation_ = false;
}

void CUDAImage::CopyToCUDAImage(CUDAImage& dst) const
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        bool needReallocation = dst.allocatedBytes_ < width_ * height_ * channels_ * GetTypeSize(elementType_);
        if(needReallocation)
        {
            dst.Allocate(width_, height_, channels_, elementType_, pitchedAllocation_);
        }
    }
    else
    {
        dst.Allocate(width_, height_, channels_, elementType_, pitchedAllocation_);
    }
    auto status = cudaMemcpy2D(dst.gpuData_, dst.pitch_, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                      height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    dst.width_ = width_;
    dst.height_ = height_;
    dst.elementType_ = elementType_;
    dst.channels_ = channels_;
}

void CUDAImage::CopyToCUDAImageAsync(DataStructures::CUDAImage& dst, void* cudaStream) const
{
    if (gpuData_ == dst.gpuData_)
    {
        return;
    }

    if(dst.gpuData_)
    {
        bool needReallocation = dst.allocatedBytes_ < width_ * height_ * channels_ * GetTypeSize(elementType_);
        if(needReallocation)
        {
            dst.AllocateAsync(width_, height_, channels_, elementType_, pitchedAllocation_, cudaStream);
        }
    }
    else
    {
        dst.AllocateAsync(width_, height_, channels_, elementType_, pitchedAllocation_, cudaStream);
    }
    auto status = cudaMemcpy2DAsync(dst.gpuData_, dst.pitch_, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                               height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data asynchronously. CUDA error " << static_cast<int>(status) << " - "
                    << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    dst.width_ = width_;
    dst.height_ = height_;
    dst.elementType_ = elementType_;
    dst.channels_ = channels_;
}

void CUDAImage::MoveFromCUDAImage(DataStructures::CUDAImage& src)
{
    if (gpuData_ == src.gpuData_)
    {
        return;
    }

    if(gpuData_)
    {
        Release();
    }

    width_ = src.width_;
    height_ = src.height_;
    pitch_ = src.pitch_;
    channels_ = src.channels_;
    elementType_ = src.elementType_;
    gpuData_ = src.gpuData_;
    allocatedBytes_ = src.allocatedBytes_;
    pitchedAllocation_ = src.pitchedAllocation_;

    src.width_ = 0;
    src.height_ = 0;
    src.pitch_ = 0;
    src.channels_ = 0;
    src.elementType_ = TYPE_UNKNOWN;
    src.gpuData_ = nullptr;
    src.allocatedBytes_= 0;
    src.pitchedAllocation_ = false;
}

void CUDAImage::MoveFromCUDAImageAsync(DataStructures::CUDAImage& src, void* cudaStream)
{
    if (gpuData_ == src.gpuData_)
    {
        return;
    }

    if(gpuData_)
    {
        ReleaseAsync(cudaStream);
    }

    width_ = src.width_;
    height_ = src.height_;
    pitch_ = src.pitch_;
    channels_ = src.channels_;
    elementType_ = src.elementType_;
    gpuData_ = src.gpuData_;
    allocatedBytes_ = src.allocatedBytes_;
    pitchedAllocation_ = src.pitchedAllocation_;

    src.width_ = 0;
    src.height_ = 0;
    src.pitch_ = 0;
    src.channels_ = 0;
    src.elementType_ = TYPE_UNKNOWN;
    src.gpuData_ = nullptr;
    src.allocatedBytes_= 0;
    src.pitchedAllocation_ = false;
}

void CUDAImage::CopyFromCUDAImage(const DataStructures::CUDAImage& src)
{
    if(gpuData_ == src.gpuData_)
    {
        return;
    }
    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < src.width_ * src.height_ * src.channels_ * GetTypeSize(src.elementType_);
        if(needReallocation)
        {
            Allocate(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_);
        }
    }
    else
    {
        Allocate(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_);
    }
    auto status = cudaMemcpy2D(gpuData_, pitch_, src.gpuData_, src.pitch_,
                               src.width_ * src.channels_ * GetTypeSize(src.elementType_), src.height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data. CUDA error " << static_cast<int>(status) << " - "
                    << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = src.width_;
    height_ = src.height_;
    channels_ = src.channels_;
    elementType_ = src.elementType_;
}

void CUDAImage::CopyFromCUDAImageAsync(const DataStructures::CUDAImage& src, void* cudaStream)
{
    if(gpuData_ == src.gpuData_)
    {
        return;
    }

    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < src.width_ * src.height_ * src.channels_ * GetTypeSize(src.elementType_);
        if(needReallocation)
        {
            AllocateAsync(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_, cudaStream);
        }
    }
    else
    {
        AllocateAsync(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_, cudaStream);
    }
    auto status = cudaMemcpy2DAsync(gpuData_, pitch_, src.gpuData_, src.pitch_, src.width_ * src.channels_ * GetTypeSize(src.elementType_),
                 src.height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data asynchronously. CUDA error " << static_cast<int>(status) << " - "
                    << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = src.width_;
    height_ = src.height_;
    channels_ = src.channels_;
    elementType_ = src.elementType_;
}

void CUDAImage::CopyFromRawDevicePointer(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation)
{
    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < width * height * channels * GetTypeSize(type);
        if(needReallocation)
        {
            Allocate(width, height, channels, type, pitchedAllocation);
        }
    }
    else
    {
        Allocate(width, height, channels, type, pitchedAllocation);
    }
    auto status = cudaMemcpy2D(gpuData_, pitch_, src, width * channels * GetTypeSize(type), width * channels * GetTypeSize(type), height,
                                    cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from raw device pointer. CUDA error " << static_cast<int>(status) << " - "
                    << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = width;
    height_ = height;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::CopyFromRawDevicePointerAsync(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream)
{
    if(gpuData_)
    {
        bool needReallocation = allocatedBytes_ < width * height * channels * GetTypeSize(type);
        if(needReallocation)
        {
            AllocateAsync(width, height, channels, type, pitchedAllocation, cudaStream);
        }
    }
    else
    {
        AllocateAsync(width, height, channels, type, pitchedAllocation, cudaStream);
    }
    auto status = cudaMemcpy2DAsync(gpuData_, pitch_, src, width * channels * GetTypeSize(type), width * channels * GetTypeSize(type), height,
                               cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data from raw device pointer asynchronously. CUDA error "
        << static_cast<int>(status) << " - " << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    width_ = width;
    height_ = height;
    channels_ = channels;
    elementType_ = type;
}

void CUDAImage::CopyToRawDevicePointer(void* dst, size_t width, size_t channels, ELEMENT_TYPE type) const
{
    if(gpuData_)
    {
        auto status = cudaMemcpy2D(dst, width * channels * GetTypeSize(type), gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                                   height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to copy image's data to raw device pointer. CUDA error "
            << static_cast<int>(status) << " - " << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    else
    {
        LOG_WARNING() << "Nothing to copy. Image is empty.";
    }
}

void CUDAImage::CopyToRawDevicePointerAsync(void* dst, size_t width, size_t channels, ELEMENT_TYPE type, void* cudaStream) const
{
    if(gpuData_)
    {
        auto status = cudaMemcpy2DAsync(dst, width * channels * GetTypeSize(type), gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                                   height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to copy image's data to raw device pointer asynchronously. CUDA error "
                        << static_cast<int>(status) << " - " << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    else
    {
        LOG_WARNING() << "Nothing to copy. Image is empty.";
    }
}

void CUDAImage::MoveToGpuMat(cv::cuda::GpuMat& dst)
{
    if (!dst.empty())
    {
        auto status = cudaFree(dst.data);

        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to release cv::cuda::GpuMat data. CUDA error "
                        << static_cast<int>(status) << " - " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst.cols = static_cast<int>(width_);
    dst.rows = static_cast<int>(height_);
    dst.step = pitch_;
    dst.flags = cv::Mat::MAGIC_VAL + (cvType & cv::Mat::TYPE_MASK);
    dst.data = gpuData_;
    dst.datastart = gpuData_;
    dst.dataend = gpuData_;

    size_t minstep = width_ * GetTypeSize(elementType_);

    dst.dataend += pitch_ * (height_ - 1) + minstep;

    width_ = 0;
    height_ = 0;
    pitch_ = 0;
    channels_ = 0;
    elementType_ = TYPE_UNKNOWN;
    gpuData_ = nullptr;
    allocatedBytes_= 0;
    pitchedAllocation_ = false;

    dst.updateContinuityFlag();
}

void CUDAImage::MoveToGpuMatAsync(cv::cuda::GpuMat& dst, void* cudaStream)
{
    if (!dst.empty())
    {
        auto status = cudaFreeAsync(dst.data, (cudaStream_t)cudaStream);

        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to release cv::cuda::GpuMat data asynchronously. CUDA error "
            << static_cast<int>(status) << " - " << cudaGetErrorName(status) << ": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst.cols = static_cast<int>(width_);
    dst.rows = static_cast<int>(height_);
    dst.step = pitch_;
    dst.flags = cv::Mat::MAGIC_VAL + (cvType & cv::Mat::TYPE_MASK);
    dst.data = gpuData_;
    dst.datastart = gpuData_;
    dst.dataend = gpuData_;

    size_t minstep = width_ * GetTypeSize(elementType_);

    dst.dataend += pitch_ * (height_ - 1) + minstep;

    width_ = 0;
    height_ = 0;
    pitch_ = 0;
    channels_ = 0;
    elementType_ = TYPE_UNKNOWN;
    gpuData_ = nullptr;
    allocatedBytes_= 0;
    pitchedAllocation_ = false;
    dst.updateContinuityFlag();
}

void CUDAImage::CopyToGpuMat(cv::cuda::GpuMat& dst) const
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    if (gpuData_ == dst.data)
    {
        return;
    }

    if(dst.data)
    {
        bool needReallocation = dst.step * dst.rows < width_ * height_ * channels_ * GetTypeSize(elementType_);
        if(needReallocation)
        {
            dst = cv::cuda::GpuMat(static_cast<int>(height_), static_cast<int>(width_), cvType);
        }
    }
    else
    {
        dst = cv::cuda::GpuMat(static_cast<int>(height_), static_cast<int>(width_), cvType);
    }
    auto status = cudaMemcpy2D(dst.data, dst.step, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                               height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data. CUDA error " << static_cast<int>(status) << " - "
                    << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    dst.cols = static_cast<int>(width_);
    dst.rows = static_cast<int>(height_);
    dst.flags = cv::Mat::MAGIC_VAL + (cvType & cv::Mat::TYPE_MASK);
    dst.updateContinuityFlag();
}

void CUDAImage::CopyToGpuMatAsync(cv::cuda::GpuMat& dst, void* cudaStream) const
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    if (gpuData_ == dst.data)
    {
        return;
    }

    if(dst.data)
    {
        bool needReallocation = dst.step * dst.rows < width_ * height_ * channels_ * GetTypeSize(elementType_);
        if(needReallocation)
        {
            dst = cv::cuda::GpuMat(static_cast<int>(height_), static_cast<int>(width_), cvType);
        }
    }
    else
    {
        dst = cv::cuda::GpuMat(static_cast<int>(height_), static_cast<int>(width_), cvType);
    }
    auto status = cudaMemcpy2DAsync(dst.data, dst.step, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                               height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data. CUDA error " << static_cast<int>(status) << " - "
                    << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }

    dst.cols = static_cast<int>(width_);
    dst.rows = static_cast<int>(height_);
    dst.flags = cv::Mat::MAGIC_VAL + (cvType & cv::Mat::TYPE_MASK);
    dst.updateContinuityFlag();
}

void CUDAImage::MoveToCvMat(cv::Mat& dst)
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), cvType);
    auto status = cudaMemcpy2D(dst.data, dst.step, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                               height_, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data to cv::Mat. CUDA error "
                    << static_cast<int>(status) << " - " << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    Release();
}

void CUDAImage::MoveToCvMatAsync(cv::Mat& dst, void* cudaStream)
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), cvType);
    auto status = cudaMemcpy2DAsync(dst.data, dst.step, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                               height_, cudaMemcpyKind::cudaMemcpyDeviceToHost, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data to cv::Mat. CUDA error "
                    << static_cast<int>(status) << " - " << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    ReleaseAsync(cudaStream);
}

void CUDAImage::CopyToCvMat(cv::Mat& dst) const
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), cvType);
    auto status = cudaMemcpy2D(dst.data, dst.step, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                                    height_, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data to cv::Mat. CUDA error "
                    << static_cast<int>(status) << " - " << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
}

void CUDAImage::CopyToCvMatAsync(cv::Mat& dst, void* cudaStream) const
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), cvType);
    auto status = cudaMemcpy2DAsync(dst.data, dst.step, gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                                    height_, cudaMemcpyKind::cudaMemcpyDeviceToHost, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to copy image's data to cv::Mat asynchronously. CUDA error "
        << static_cast<int>(status) << " - " << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
}

bool CUDAImage::Allocated() const
{
    return gpuData_ != nullptr;
}

size_t CUDAImage::GetElementSize()  const
{
    return GetTypeSize(elementType_);
}

}
