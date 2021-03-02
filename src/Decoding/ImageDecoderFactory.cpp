#include "ImageDecoderFactory.h"
#include "OpenCVImageDecoder.h"
#include "Logger.h"

#include "NvJPEGImageDecoder.h"

namespace Decoding
{

std::unique_ptr<IImageDecoder> ImageDecoderFactory::Create()
{

    if (auto decoder = std::make_unique<NvJPEGImageDecoder>(); decoder->IsInitialized())
    {
        return decoder;
    }

    return std::make_unique<OpenCVImageDecoder>();
}

}