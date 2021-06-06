/**
 * @file KeyPointMatchingAlgorithm.h.
 *
 * @brief
 */

#ifndef KEY_POINT_MATCHING_ALGORITHM_H
#define KEY_POINT_MATCHING_ALGORITHM_H

#include "ICPUAlgorithm.h"


namespace openMVG::matching_image_collection
{
    class Matcher;
}

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
class KeyPointMatchingAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    KeyPointMatchingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~KeyPointMatchingAlgorithm() override;

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
    float distanceRatio_;

    ///
    std::unique_ptr<openMVG::matching_image_collection::Matcher> matcher_;

};

}

#endif // KEY_POINT_MATCHING_ALGORITHM_H
