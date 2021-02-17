#ifndef IMAGE_DECODER_FACTORY_H
#define IMAGE_DECODER_FACTORY_H

#include <memory>

class IImageDecoder;

class ImageDecoderFactory
{
public:
    static std::unique_ptr<IImageDecoder> Create();
};

#endif // IMAGE_DECODER_FACTORY_H