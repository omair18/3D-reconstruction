#ifndef IMAGE_DECODING_ALGORITHM_H
#define IMAGE_DECODING_ALGORITHM_H

#include <vector>
#include <memory>

namespace Decoding
{
    class IImageDecoder;
}

class ImageDecodingAlgorithm
{

public:

private:

    std::vector<std::unique_ptr<Decoding::IImageDecoder>> decoders_;

};


#endif // IMAGE_DECODING_ALGORITHM_H
