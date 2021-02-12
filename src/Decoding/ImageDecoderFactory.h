#pragma once
#include <memory>

class IImageDecoder;

class ImageDecoderFactory
{
public:
    static std::unique_ptr<IImageDecoder> Create();
};

