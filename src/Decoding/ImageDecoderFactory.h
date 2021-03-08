#ifndef IMAGE_DECODER_FACTORY_H
#define IMAGE_DECODER_FACTORY_H

#include <memory>

namespace Decoding
{

class IImageDecoder;

enum DecoderType
{
    OPENCV_IMAGE_DECODER = 0,
    NVJPEG_IMAGE_DECODER,
    NVJPEG_HARDWARE_IMAGE_DECODER,
    NVJPEG2K_IMAGE_DECODER
};

class ImageDecoderFactory final
{
public:
    static std::unique_ptr<IImageDecoder> Create(DecoderType type, bool useCUDAStream, void* cudaStream);
};

}

#endif // IMAGE_DECODER_FACTORY_H