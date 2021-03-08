#include "ImageDecoderFactory.h"
#include "OpenCVImageDecoder.h"
#include "Logger.h"
#include "NvJPEGImageDecoder.h"
#include "NvJPEG2kImageDecoder.h"
#include "NvJPEGHardwareImageDecoder.h"

namespace Decoding
{

std::unique_ptr<IImageDecoder> ImageDecoderFactory::Create(DecoderType type, bool useCUDAStream, void *cudaStream)
{
    switch (type)
    {
        case DecoderType::OPENCV_IMAGE_DECODER:
        {
            LOG_TRACE() << "Creating OpenCV image decoder ...";
            if(useCUDAStream)
            {
                LOG_WARNING() << "OpenCV image decoder doesn't use CUDA stream.";
            }
        } break;

        case DecoderType::NVJPEG_IMAGE_DECODER:
        {
            LOG_TRACE() << "Creating NvJPEG image decoder ...";
            if(!useCUDAStream)
            {
                LOG_ERROR() << "NvJPEG image decoder needs CUDA stream.";
                return nullptr;
            }
        } break;

        case DecoderType::NVJPEG_HARDWARE_IMAGE_DECODER:
        {
            LOG_TRACE() << "Creating NvJPEG hardware image decoder ...";
            if(!useCUDAStream)
            {
                LOG_ERROR() << "NvJPEG hardware image decoder needs CUDA stream.";
                return nullptr;
            }
        } break;

        case DecoderType::NVJPEG2K_IMAGE_DECODER:
        {
            LOG_TRACE() << "Creating NvJPEG2K image decoder ...";
            if(!useCUDAStream)
            {
                LOG_ERROR() << "NvJPEG2K image decoder needs CUDA stream.";
                return nullptr;
            }
        } break;

        default:
        {
            LOG_ERROR() << "Unknown decoder type.";
            return nullptr;
        }
    }
}

}