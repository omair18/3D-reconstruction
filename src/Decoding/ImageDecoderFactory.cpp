#include "ImageDecoderFactory.h"
#include "OpenCVImageDecoder.h"
#include "Logger.h"

#include "NvJPEGDecoder.h"


std::unique_ptr<IImageDecoder> ImageDecoderFactory::Create()
{

    if (auto decoder = std::make_unique<NvJPEGDecoder>(); decoder->IsInitialized())
    {
        return decoder;
    }

    return std::make_unique<OpenCVImageDecoder>();
}