/**
 * @file CUDAKeyPointFilteringAlgorithm.h.
 *
 * @brief
 */

#ifndef CUDA_KEY_POINT_FILTERING_ALGORITHM_H
#define CUDA_KEY_POINT_FILTERING_ALGORITHM_H

#include "IGPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class CUDAKeyPointFilteringAlgorithm
 *
 * @brief
 */
class CUDAKeyPointFilteringAlgorithm : public IGPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    CUDAKeyPointFilteringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream);

    /**
     * @brief
     */
    ~CUDAKeyPointFilteringAlgorithm() override = default;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData) override;

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
    void* cudaStream_;

};

}

#endif // CUDA_KEY_POINT_FILTERING_ALGORITHM_H
