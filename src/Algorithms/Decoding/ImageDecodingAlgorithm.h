/**
 * @file ImageDecodingAlgorithm.h.
 *
 * @brief
 */

#ifndef IMAGE_DECODING_ALGORITHM_H
#define IMAGE_DECODING_ALGORITHM_H

#include <vector>

#include "ICPUAlgorithm.h"

// // forward declaration for Decoding::IImageDecoder
namespace Decoding
{
    class IImageDecoder;
}

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class ImageDecodingAlgorithm
 *
 * @brief
 */
class ImageDecodingAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    ImageDecodingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                               [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                               [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~ImageDecodingAlgorithm() override;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    /**
     * @brief
     *
     * @param config
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    ///
    std::vector<std::shared_ptr<Decoding::IImageDecoder>> decoders_;

};

}

#endif // IMAGE_DECODING_ALGORITHM_H
