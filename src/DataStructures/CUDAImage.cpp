#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

#include "CUDAImage.h"
#include "Logger.h"

namespace DataStructures
{

size_t GetTypeSize(CUDAImage::ELEMENT_TYPE type)
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

std::pair<CUDAImage::ELEMENT_TYPE, size_t> ConvertCvTypeToCUDAImageElementTypeAndChannels(int type)
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

int ConvertCUDAImageElementTypeAndChannelsToCvType(CUDAImage::ELEMENT_TYPE type, size_t channels)
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

double BytesToMB(size_t bytes)
{
    return bytes / 1024. / 1024.;
}

DataStructures::CUDAImage::CUDAImage() :
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

void DataStructures::CUDAImage::AllocateAsync(unsigned int width, unsigned int height, unsigned int channels, ELEMENT_TYPE type, bool pitchedAllocation, void* cudaStream)
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

void DataStructures::CUDAImage::Release()
{
    auto status = cudaFree(gpuData_);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to release " << (allocatedBytes_ / 1024. / 1024.) << " MB of GPU memory. "
        << "CUDA error " << static_cast<int>(status) << " - " << cudaGetErrorName(status)
        << ": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    else
    {
        LOG_TRACE() << "Successfully released " << (allocatedBytes_ / 1024. / 1024.) << " MB of GPU memory." ;
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

void DataStructures::CUDAImage::ReleaseAsync(void* cudaStream)
{
    auto status = cudaFreeAsync(gpuData_, (cudaStream_t)cudaStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << "Failed to release " << (allocatedBytes_ / 1024. / 1024.) << " MB of GPU memory asynchronously. "
                    << "CUDA error " << static_cast<int>(status) << " - " << cudaGetErrorName(status)
                    << ": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    else
    {
        LOG_TRACE() << "Successfully released " << (allocatedBytes_ / 1024. / 1024.) << " MB of GPU memory asynchronously.";
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

DataStructures::CUDAImage::CUDAImage(const DataStructures::CUDAImage &other)
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

DataStructures::CUDAImage::CUDAImage(DataStructures::CUDAImage &&other) noexcept
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
    if (this == reinterpret_cast<const CUDAImage*>(&other))
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
    {
        return *this;
    }
    CopyFromCvMat(other);
    return *this;
}

DataStructures::CUDAImage &DataStructures::CUDAImage::operator=(cv::Mat &&other) noexcept
{
    MoveFromCvMat(other);
    return *this;
}

bool DataStructures::CUDAImage::operator==(const DataStructures::CUDAImage &other) const
{
    return width_ == other.width_ &&
         height_ == other.height_ &&
         pitch_ == other.pitch_ &&
         channels_ == other.channels_ &&
         elementType_ == other.elementType_;
}

bool DataStructures::CUDAImage::operator==(const cv::cuda::GpuMat& other) const
{
    return width_ == static_cast<size_t>(other.cols) &&
           height_ == static_cast<size_t>(other.rows) &&
           pitch_ == other.step &&
           channels_ == static_cast<size_t>(other.channels()) &&
           GetTypeSize(elementType_) == other.elemSize1();
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

void DataStructures::CUDAImage::CopyFromGpuMat(const cv::cuda::GpuMat &src)
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
}

void DataStructures::CUDAImage::MoveFromCvMat(cv::Mat &src)
{
    CopyFromCvMat(src);
    src.release();
}

void DataStructures::CUDAImage::CopyFromCvMat(const cv::Mat &src)
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
}

bool DataStructures::CUDAImage::Empty() const
{
    return gpuData_ == nullptr;
}

void DataStructures::CUDAImage::CopyFromRawHostPointer(void* src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation)
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

    auto [type, channels] = ConvertCvTypeToCUDAImageElementTypeAndChannels(src.type());

    std::swap(gpuData_, src.data);
    src.data = nullptr;
    src.datastart = nullptr;
    src.dataend = nullptr;

    width_ = src.cols;
    src.cols = 0;

    height_ = src.rows;
    src.rows = 0;

    channels_ = channels;

    elementType_ = type;

    pitch_ = src.step;
    src.step = 0;

    src.flags = 0;

    pitchedAllocation_ = width_ * channels_ * GetTypeSize(type) != pitch_;
    allocatedBytes_ = pitch_ * height_;
}

void DataStructures::CUDAImage::CopyFromGpuMatAsync(const cv::cuda::GpuMat &src, void *cudaStream)
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
        LOG_ERROR() << "Failed to copy image's data from cv::cuda::GpuMat asyncronously. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
}

void DataStructures::CUDAImage::MoveFromCvMatAsync(cv::Mat &src, void *cudaStream)
{
    CopyFromCvMatAsync(src, cudaStream);
    src.release();
}

void DataStructures::CUDAImage::CopyFromCvMatAsync(const cv::Mat &src, void *cudaStream)
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
        LOG_ERROR() << "Failed to copy image's data from cv::Mat asyncronously. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
}

void DataStructures::CUDAImage::CopyFromRawHostPointerAsync(void *src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void *cudaStream)
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
        LOG_ERROR() << "Failed to copy image's data from raw host pointer asyncronously. CUDA error " << static_cast<int>(status) << " - "
        << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
}

void DataStructures::CUDAImage::CopyToRawHostPointer(void *dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type)
{
    if(gpuData_)
    {
        auto status = cudaMemcpy2D(dst, width * channels * GetTypeSize(type), gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                     height_, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to copy image's data to raw host. CUDA error " << static_cast<int>(status) << " - "
            << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }
    else
    {
        LOG_WARNING() << "Nothing to copy. Image is empty.";
    }
}

void DataStructures::CUDAImage::CopyToRawHostPointerAsync(void *dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, void *cudaStream)
{
    if(gpuData_)
    {
        cudaMemcpy2DAsync(dst, width * channels * GetTypeSize(type), gpuData_, pitch_, width_ * channels_ * GetTypeSize(elementType_),
                     height_, cudaMemcpyKind::cudaMemcpyDeviceToHost, (cudaStream_t)cudaStream);
    }
    else
    {
        LOG_WARNING() << "Nothing to copy. Image is empty.";
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
    std::swap(dst.height_, height_);
    std::swap(dst.pitch_, pitch_);
    std::swap(dst.channels_, channels_);
    std::swap(dst.elementType_, elementType_);
    std::swap(dst.gpuData_, gpuData_);
    std::swap(dst.allocatedBytes_, allocatedBytes_);
    std::swap(dst.pitchedAllocation_, pitchedAllocation_);

    width_ = 0;
    height_ = 0;
    pitch_ = 0;
    channels_ = 0;
    elementType_ = TYPE_UNKNOWN;
    gpuData_ = nullptr;
    allocatedBytes_ = 0;
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
    std::swap(dst.height_, height_);
    std::swap(dst.pitch_, pitch_);
    std::swap(dst.channels_, channels_);
    std::swap(dst.elementType_, elementType_);
    std::swap(dst.gpuData_, gpuData_);
    std::swap(dst.allocatedBytes_, allocatedBytes_);
    std::swap(dst.pitchedAllocation_, pitchedAllocation_);

    width_ = 0;
    height_ = 0;
    pitch_ = 0;
    channels_ = 0;
    elementType_ = TYPE_UNKNOWN;
    gpuData_ = nullptr;
    allocatedBytes_ = 0;
    pitchedAllocation_ = false;
}

void DataStructures::CUDAImage::CopyToCUDAImage(DataStructures::CUDAImage &dst) const
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

void DataStructures::CUDAImage::CopyToCUDAImageAsync(DataStructures::CUDAImage &dst, void* cudaStream) const
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
        LOG_ERROR() << "Failed to copy image's data asyncronously. CUDA error " << static_cast<int>(status) << " - "
                    << cudaGetErrorName(status) <<": " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA error.");
    }
    dst.width_ = width_;
    dst.height_ = height_;
    dst.elementType_ = elementType_;
    dst.channels_ = channels_;
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
    std::swap(height_, src.height_);
    std::swap(pitch_, src.pitch_);
    std::swap(channels_, src.channels_);
    std::swap(elementType_, src.elementType_);
    std::swap(gpuData_, src.gpuData_);
    std::swap(allocatedBytes_, src.allocatedBytes_);
    std::swap(pitchedAllocation_, src.pitchedAllocation_);

    src.width_ = 0;
    src.height_ = 0;
    src.pitch_ = 0;
    src.channels_ = 0;
    src.elementType_ = TYPE_UNKNOWN;
    src.gpuData_ = nullptr;
    src.allocatedBytes_= 0;
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
    std::swap(height_, src.height_);
    std::swap(pitch_, src.pitch_);
    std::swap(channels_, src.channels_);
    std::swap(elementType_, src.elementType_);
    std::swap(gpuData_, src.gpuData_);
    std::swap(allocatedBytes_, src.allocatedBytes_);
    std::swap(pitchedAllocation_, src.pitchedAllocation_);

    src.width_ = 0;
    src.height_ = 0;
    src.pitch_ = 0;
    src.channels_ = 0;
    src.elementType_ = TYPE_UNKNOWN;
    src.gpuData_ = nullptr;
    src.allocatedBytes_= 0;
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
                GetTypeSize(elementType_) != GetTypeSize(src.elementType_);
        if(needReallocation)
        {
            Allocate(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_);
        }
    }
    else
    {
        Allocate(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_);
    }
    cudaMemcpy2D(gpuData_, pitch_, src.gpuData_, src.pitch_, src.width_ * src.channels_ * GetTypeSize(src.elementType_),
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
                GetTypeSize(elementType_) != GetTypeSize(src.elementType_);
        if(needReallocation)
        {
            AllocateAsync(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_, cudaStream);
        }
    }
    else
    {
        AllocateAsync(src.width_, src.height_, src.channels_, src.elementType_, src.pitchedAllocation_, cudaStream);
    }
    cudaMemcpy2DAsync(gpuData_, pitch_, src.gpuData_, src.pitch_, src.width_ * src.channels_ * GetTypeSize(src.elementType_),
                 src.height_, cudaMemcpyKind::cudaMemcpyDeviceToDevice, (cudaStream_t)cudaStream);
}

void DataStructures::CUDAImage::CopyFromRawDevicePointer(void *src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation)
{

}

void DataStructures::CUDAImage::CopyFromRawDevicePointerAsync(void *src, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, bool pitchedAllocation, void *cudaStream)
{

}

void DataStructures::CUDAImage::CopyToRawDevicePointer(void *dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type)
{

}

void DataStructures::CUDAImage::CopyToRawDevicePointerAsync(void *dst, size_t width, size_t height, size_t channels, ELEMENT_TYPE type, void *cudaStream)
{

}

void CUDAImage::MoveToGpuMat(cv::cuda::GpuMat &dst)
{

}

void CUDAImage::MoveToGpuMatAsync(cv::cuda::GpuMat &dst, void *cudaStream)
{

}

void CUDAImage::CopyToGpuMat(cv::cuda::GpuMat &dst)
{

}

void CUDAImage::CopyToGpuMatAsync(cv::cuda::GpuMat &dst, void *cudaStream)
{

}

void CUDAImage::MoveToCvMat(cv::Mat &dst)
{

}

void CUDAImage::MoveToCvMatAsync(cv::Mat &dst, void *cudaStream)
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst = cv::Mat(height_, width_, cvType);
    gpuData_ = nullptr;
}

void CUDAImage::CopyToCvMat(cv::Mat &dst) const
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst = cv::Mat(height_, width_, cvType);
}

void CUDAImage::CopyToCvMatAsync(cv::Mat &dst, void *cudaStream) const
{
    auto cvType = ConvertCUDAImageElementTypeAndChannelsToCvType(elementType_, channels_);
    dst = cv::Mat(height_, width_, cvType);

}

}
