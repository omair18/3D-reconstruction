/**
 * @file ImageDecoderFactory.h.
 *
 * @brief Declares class of ImageDecoderFactory. This class is used to create different types of image decoders.
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

// forward declaration for Decoding::IImageDecoder
class IImageDecoder;

/**
 * @enum DecoderType
 *
 * @brief Enum of different decoders types.
 */
enum DecoderType
{
    /// Image decoder with OpenCV backend (CPU)
    OPENCV_IMAGE_DECODER = 0,

    /// Image decoder with NvJPEG backend (GPU)
    NVJPEG_IMAGE_DECODER,

    /// Image decoder with NvJPEG hardware backend (GPU)
    NVJPEG_HARDWARE_IMAGE_DECODER,

    /// Image decoder with NvJPEG2000 backend (GPU)
    NVJPEG2K_IMAGE_DECODER,

    /// Unknown decoder
    UNKNOWN_DECODER
};

/**
 * @class ImageDecoderFactory
 *
 * @brief This class is used to create different types of image decoders.
 */
class ImageDecoderFactory final
{
public:

    /**
     * @brief Creates a pointer to a specific image decoder.
     *
     * @param type - Type of image decoder to be created
     * @param useCUDAStream - Flag of usage CUDA stream. If decoder uses CUDA stream, this flag must be set to true
     * @param cudaStream - CUDA stream to use
     * @return Pointer to a specific image decoder.
     */
    static std::unique_ptr<IImageDecoder> Create(DecoderType type, bool useCUDAStream, void* cudaStream = nullptr);
};

}

#endif // IMAGE_DECODER_FACTORY_H