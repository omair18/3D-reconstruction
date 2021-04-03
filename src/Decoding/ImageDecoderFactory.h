/**
 * @file ImageDecoderFactory.h.
 *
 * @brief
 */

#ifndef IMAGE_DECODER_FACTORY_H
#define IMAGE_DECODER_FACTORY_H

#include <memory>

/**
 * @namespace Decoding
 *
 * @brief Namespace of libdecoding library.
 */
namespace Decoding
{

/**
 * @class IImageDecoder
 *
 * @brief
 */
class IImageDecoder;

/**
 * @enum DecoderType
 *
 * @brief
 */
enum DecoderType
{
    ///
    OPENCV_IMAGE_DECODER = 0,

    ///
    NVJPEG_IMAGE_DECODER,

    ///
    NVJPEG_HARDWARE_IMAGE_DECODER,

    ///
    NVJPEG2K_IMAGE_DECODER
};

/**
 * @class ImageDecoderFactory
 *
 * @brief
 */
class ImageDecoderFactory final
{
public:

    /**
     * @brief
     *
     * @param type
     * @param useCUDAStream
     * @param cudaStream
     * @return
     */
    static std::unique_ptr<IImageDecoder> Create(DecoderType type, bool useCUDAStream, void* cudaStream = nullptr);
};

}

#endif // IMAGE_DECODER_FACTORY_H