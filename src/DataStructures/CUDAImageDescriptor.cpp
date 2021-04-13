#include "CUDAImageDescriptor.h"

namespace DataStructures
{

CUDAImageDescriptor::CUDAImageDescriptor() :
image_(),
frameId_(0),
cameraId_(0),
timestamp_(0),
focalLength_(0),
sensorSize_(0),
rawImageData_()
{

}

CUDAImageDescriptor::CUDAImageDescriptor(CUDAImageDescriptor &&other) noexcept
{
    image_ = std::move(other.image_);
    frameId_ = other.frameId_;
    cameraId_ = other.cameraId_;
    timestamp_ = other.timestamp_;
    focalLength_ = other.focalLength_;
    sensorSize_ = other.sensorSize_;
    rawImageData_ = std::move(other.rawImageData_);
}

bool CUDAImageDescriptor::operator==(const CUDAImageDescriptor &other)
{
    return false;
}

CUDAImageDescriptor &CUDAImageDescriptor::operator=(CUDAImageDescriptor &&other) noexcept
{
    image_ = std::move(other.image_);
    frameId_ = other.frameId_;
    cameraId_ = other.cameraId_;
    timestamp_ = other.timestamp_;
    focalLength_ = other.focalLength_;
    sensorSize_ = other.sensorSize_;
    rawImageData_ = std::move(other.rawImageData_);
    return *this;
}

const CUDAImage &CUDAImageDescriptor::GetCUDAImage()
{
    return image_;
}

void CUDAImageDescriptor::SetCUDAImage(const CUDAImage &image)
{
    image_ = image;
}

void CUDAImageDescriptor::SetCUDAImage(CUDAImage &&image) noexcept
{
    image_ = std::move(image);
}

int CUDAImageDescriptor::GetFrameId() const noexcept
{
    return frameId_;
}

void CUDAImageDescriptor::SetFrameId(int frameId)
{
    frameId_ = frameId;
}

int CUDAImageDescriptor::GetCameraId() const noexcept
{
    return cameraId_;
}

void CUDAImageDescriptor::SetCameraId(int cameraId)
{
    cameraId_ = cameraId;
}

unsigned long CUDAImageDescriptor::GetTimestamp() const noexcept
{
    return timestamp_;
}

void CUDAImageDescriptor::SetTimestamp(unsigned long timestamp)
{
    timestamp_ = timestamp;
}

float CUDAImageDescriptor::GetFocalLength() const noexcept
{
    return focalLength_;
}

void CUDAImageDescriptor::SetFocalLength(float focalLength)
{
    focalLength_ = focalLength;
}

float CUDAImageDescriptor::GetSensorSize() const noexcept
{
    return sensorSize_;
}

void CUDAImageDescriptor::SetSensorSize(float sensorSize)
{
    sensorSize_ = sensorSize;
}

const std::vector<unsigned char> &CUDAImageDescriptor::GetRawImageData() noexcept
{
    return rawImageData_;
}

void CUDAImageDescriptor::SetRawImageData(const std::vector<unsigned char> &rawImageData)
{
    rawImageData_ = rawImageData;
}

void CUDAImageDescriptor::SetRawImageData(std::vector<unsigned char> &&rawImageData) noexcept
{
    rawImageData_ = std::move(rawImageData);
}

void CUDAImageDescriptor::ClearRawImageData() noexcept
{
    rawImageData_.clear();
}

}