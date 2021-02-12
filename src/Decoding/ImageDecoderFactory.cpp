#include "ImageDecoderFactory.h"
#include "OpenCVImageDecoder.h"
#include "Logger.h"

#ifdef WITH_NVJPEG
#include "NvidiaImageDecoder.h"
#endif

std::unique_ptr<IImageDecoder> ImageDecoderFactory::Create()
{
#ifdef WITH_NVJPEG
    LOG_TRACE("Create GPU decoder");

    if (auto decoder = std::make_unique<NvidiaImageDecoder>(); decoder->IsInitialized())
    {
        return decoder;
    }

    LOG_WARNING("Failed to create GPU decoder");
#endif
    LOG_TRACE("Create CPU decoder");

    return std::make_unique<OpenCVImageDecoder>();
}