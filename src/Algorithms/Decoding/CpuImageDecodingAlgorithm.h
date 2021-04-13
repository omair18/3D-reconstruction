/**
 * @file CpuImageDecodingAlgorithm.h.
 *
 * @brief
 */

#ifndef CPU_IMAGE_DECODING_ALGORITHM_H
#define CPU_IMAGE_DECODING_ALGORITHM_H

#include <vector>

#include "ICPUAlgorithm.h"

// forward declaration for Decoding::IImageDecoder
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
 * @class CpuImageDecodingAlgorithm
 *
 * @brief
 */
class CpuImageDecodingAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    CpuImageDecodingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                              [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                              [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~CpuImageDecodingAlgorithm() override = default;

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

    /**
     * @brief
     *
     * @param config
     */
    static void ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config);

    /**
     * @brief
     *
     * @param config
     */
    void InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config);

    ///
    std::vector<std::shared_ptr<Decoding::IImageDecoder>> decoders_;

    bool removeSourceData_;

};

}

#endif // CPU_IMAGE_DECODING_ALGORITHM_H
