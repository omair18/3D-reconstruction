#include <opencv2/core/mat.hpp>

#include "ImageDescriptor.h"
#include "CUDAImage.h"

namespace DataStructures
{

ImageDescriptor::ImageDescriptor() :
CUDAImage_(std::make_unique<CUDAImage>()),
hostImage_(std::make_unique<cv::Mat>()),
location_(LOCATION::UNDEFINED),
distortionFunctionId_(0),
frameId_(0),
cameraId_(0),
timestamp_(0),
focalLength_(0),
sensorSize_(0),
rawImageData_()
{

}

ImageDescriptor::ImageDescriptor(ImageDescriptor&& other) noexcept
{
    CUDAImage_ = std::move(other.CUDAImage_);
    hostImage_ = std::move(other.hostImage_);
    location_ = other.location_;
    frameId_ = other.frameId_;
    cameraId_ = other.cameraId_;
    timestamp_ = other.timestamp_;
    focalLength_ = other.focalLength_;
    sensorSize_ = other.sensorSize_;
    rawImageData_ = std::move(other.rawImageData_);
    distortionFunctionId_ = other.distortionFunctionId_;

    other.location_ = LOCATION::UNDEFINED;
    other.frameId_ = 0;
    other.cameraId_ = 0;
    other.timestamp_ = 0;
    other.focalLength_ = 0;
    other.sensorSize_ = 0;
    other.distortionFunctionId_ = 0;
}

ImageDescriptor::ImageDescriptor(const ImageDescriptor &other)
{
    location_ = other.location_;
    switch (location_)
    {
        case LOCATION::HOST:
        {
            if (other.hostImage_)
            {
                hostImage_ = std::make_unique<cv::Mat>(*other.hostImage_);
                CUDAImage_ = nullptr;
            }
        } break;

        case LOCATION::DEVICE:
        {
            if (other.CUDAImage_)
            {
                CUDAImage_ = std::make_unique<CUDAImage>(*other.CUDAImage_);
                hostImage_ = nullptr;
            }
        } break;

        default: break;
    }
    frameId_ = other.frameId_;
    cameraId_ = other.cameraId_;
    timestamp_ = other.timestamp_;
    focalLength_ = other.focalLength_;
    sensorSize_ = other.sensorSize_;
    rawImageData_ = (other.rawImageData_);
    distortionFunctionId_ = other.distortionFunctionId_;
}

ImageDescriptor::~ImageDescriptor() = default;

ImageDescriptor& ImageDescriptor::operator=(ImageDescriptor&& other) noexcept
{
    location_ = other.location_;
    CUDAImage_ = std::move(other.CUDAImage_);
    hostImage_ = std::move(other.hostImage_);
    frameId_ = other.frameId_;
    cameraId_ = other.cameraId_;
    timestamp_ = other.timestamp_;
    focalLength_ = other.focalLength_;
    sensorSize_ = other.sensorSize_;
    rawImageData_ = std::move(other.rawImageData_);
    distortionFunctionId_ = other.distortionFunctionId_;

    other.location_ = LOCATION::UNDEFINED;
    other.frameId_ = 0;
    other.cameraId_ = 0;
    other.timestamp_ = 0;
    other.focalLength_ = 0;
    other.distortionFunctionId_ = 0;
    other.sensorSize_ = 0;
    return *this;
}

ImageDescriptor& ImageDescriptor::operator=(const ImageDescriptor& other)
{
    location_ = other.location_;
    switch (location_)
    {
        case LOCATION::HOST:
        {
            if (other.hostImage_)
            {
                if (hostImage_)
                {
                    *hostImage_ = *other.hostImage_;
                }
                else
                {
                    hostImage_ = std::make_unique<cv::Mat>(*other.hostImage_);
                }
                CUDAImage_ = nullptr;
            }
        } break;

        case LOCATION::DEVICE:
        {
            if (other.CUDAImage_)
            {
                if (CUDAImage_)
                {
                    *CUDAImage_ = *other.CUDAImage_;
                }
                else
                {
                    CUDAImage_ = std::make_unique<CUDAImage>(*other.CUDAImage_);
                }
                hostImage_ = nullptr;
            }
        } break;

        default: break;
    }
    frameId_ = other.frameId_;
    cameraId_ = other.cameraId_;
    timestamp_ = other.timestamp_;
    focalLength_ = other.focalLength_;
    sensorSize_ = other.sensorSize_;
    rawImageData_ = other.rawImageData_;
    distortionFunctionId_ = other.distortionFunctionId_;
    return *this;
}

const std::unique_ptr<CUDAImage>& ImageDescriptor::GetCUDAImage() const
{
    return CUDAImage_;
}

void ImageDescriptor::SetCUDAImage(const CUDAImage& image)
{
    if (CUDAImage_)
    {
        *CUDAImage_ = image;
    }
    else
    {
        CUDAImage_ = std::make_unique<CUDAImage>(image);
    }
}

void ImageDescriptor::SetCUDAImage(CUDAImage&& image) noexcept
{
    if (CUDAImage_)
    {
        *CUDAImage_ = std::move(image);
    }
    else
    {
        CUDAImage_ = std::make_unique<CUDAImage>(std::move(image));
    }
}

void ImageDescriptor::SetCUDAImage(const std::unique_ptr<CUDAImage>& image)
{
    if (image)
    {
        if (CUDAImage_)
        {
            *CUDAImage_ = *image;
        }
        else
        {
            CUDAImage_ = std::make_unique<CUDAImage>(*image);
        }
    }
    else
    {
        CUDAImage_ = nullptr;
    }
}

void ImageDescriptor::SetCUDAImage(std::unique_ptr<CUDAImage>&& image) noexcept
{
    if (image)
    {
        if (CUDAImage_)
        {
            *CUDAImage_ = std::move(*image);
        }
        else
        {
            CUDAImage_ = std::make_unique<CUDAImage>(std::move(*image));
        }
    }
    else
    {
        CUDAImage_ = nullptr;
    }
}

int ImageDescriptor::GetFrameId() const noexcept
{
    return frameId_;
}

void ImageDescriptor::SetFrameId(int frameId)
{
    frameId_ = frameId;
}

int ImageDescriptor::GetCameraId() const noexcept
{
    return cameraId_;
}

void ImageDescriptor::SetCameraId(int cameraId)
{
    cameraId_ = cameraId;
}

unsigned long ImageDescriptor::GetTimestamp() const noexcept
{
    return timestamp_;
}

void ImageDescriptor::SetTimestamp(unsigned long timestamp)
{
    timestamp_ = timestamp;
}

float ImageDescriptor::GetFocalLength() const noexcept
{
    return focalLength_;
}

void ImageDescriptor::SetFocalLength(float focalLength)
{
    focalLength_ = focalLength;
}

float ImageDescriptor::GetSensorSize() const noexcept
{
    return sensorSize_;
}

void ImageDescriptor::SetSensorSize(float sensorSize)
{
    sensorSize_ = sensorSize;
}

const std::vector<unsigned char>& ImageDescriptor::GetRawImageData() const noexcept
{
    return rawImageData_;
}

void ImageDescriptor::SetRawImageData(const std::vector<unsigned char>& rawImageData)
{
    rawImageData_ = rawImageData;
}

void ImageDescriptor::SetRawImageData(std::vector<unsigned char>&& rawImageData) noexcept
{
    rawImageData_ = std::move(rawImageData);
}

void ImageDescriptor::ClearRawImageData() noexcept
{
    rawImageData_.clear();
}

void ImageDescriptor::SetHostImage(const cv::Mat &image)
{
    if (hostImage_)
    {
        *hostImage_ = std::move(image.clone());
    }
    else
    {
        hostImage_ = std::make_unique<cv::Mat>(std::move(image.clone()));
    }
}

void ImageDescriptor::SetHostImage(cv::Mat &&image)
{
    if (hostImage_)
    {
        *hostImage_ = std::move(image);
    }
    else
    {
        hostImage_ = std::make_unique<cv::Mat>(std::move(image));
    }
}

void ImageDescriptor::SetHostImage(const std::unique_ptr<cv::Mat> &image)
{
    if (image)
    {
        if (hostImage_)
        {
            *hostImage_ = *image;
        }
        else
        {
            hostImage_ = std::make_unique<cv::Mat>(*image);
        }
    }
    else
    {
        hostImage_ = nullptr;
    }
}

void ImageDescriptor::SetHostImage(std::unique_ptr<cv::Mat> &&image)
{
    if (image)
    {
        if (hostImage_)
        {
            *hostImage_ = std::move(*image);
        }
        else
        {
            hostImage_ = std::make_unique<cv::Mat>(std::move(*image));
        }
    }
    else
    {
        hostImage_ = nullptr;
    }
}

const std::unique_ptr<cv::Mat>& ImageDescriptor::GetHostImage() const noexcept
{
    return hostImage_;
}

void ImageDescriptor::SetDataLocation(ImageDescriptor::LOCATION location)
{
    location_ = location;
}

ImageDescriptor::LOCATION ImageDescriptor::GetDataLocation() const noexcept
{
    return location_;
}

void ImageDescriptor::SetCameraDistortionFunctionId(int id)
{
    distortionFunctionId_ = id;
}

int ImageDescriptor::GetCameraDistortionFunctionId() const noexcept
{
    return distortionFunctionId_;
}

}