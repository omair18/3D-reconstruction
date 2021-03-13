#include "CUDAImageDescriptor.h"

namespace DataStructures
{

CUDAImageDescriptor::CUDAImageDescriptor()
{

}

CUDAImageDescriptor::~CUDAImageDescriptor()
{

}

bool CUDAImageDescriptor::operator==(const CUDAImageDescriptor &other)
{
    return false;
}

CUDAImageDescriptor &CUDAImageDescriptor::operator=(const CUDAImageDescriptor &other)
{
    return *this;
}

CUDAImageDescriptor &CUDAImageDescriptor::operator=(CUDAImageDescriptor &&other) noexcept
{
    return *this;
}

const CUDAImage &CUDAImageDescriptor::GetCUDAImage()
{
    return image_;
}

void CUDAImageDescriptor::SetCUDAImage(const CUDAImage &image)
{

}

void CUDAImageDescriptor::SetCUDAImage(CUDAImage &&image) noexcept
{

}

int CUDAImageDescriptor::GetFrameId() noexcept
{
    return 0;
}

void CUDAImageDescriptor::SetFrameId(int frameId)
{

}

int CUDAImageDescriptor::GetCameraId() noexcept
{
    return 0;
}

void CUDAImageDescriptor::SetCameraId(int cameraId)
{

}

int CUDAImageDescriptor::GetTimestamp() noexcept
{
    return 0;
}

void CUDAImageDescriptor::SetTimestamp(int timestamp)
{

}

}