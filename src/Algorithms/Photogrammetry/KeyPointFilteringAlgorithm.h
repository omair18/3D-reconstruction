/**
 * @file KeyPointFilteringAlgorithm.h.
 *
 * @brief
 */

#ifndef KEY_POINT_FILTERING_ALGORITHM_H
#define KEY_POINT_FILTERING_ALGORITHM_H

#include "ICPUAlgorithm.h"

class FundamentalMatrixRobustModelEstimator;

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class KeyPointMatchingAlgorithm
 *
 * @brief
 */
class KeyPointFilteringAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    KeyPointFilteringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~KeyPointFilteringAlgorithm() override;

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
    int maxIterations_;

    ///
    float estimationPrecision_;

    ///
    bool isInitialized_;

    std::unique_ptr<FundamentalMatrixRobustModelEstimator> modelEstimator_;

};

}

#endif // KEY_POINT_FILTERING_ALGORITHM_H
