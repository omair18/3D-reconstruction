/**
 * @file AKAZEKeypointDetectionAlgorithm.h.
 *
 * @brief
 */

#ifndef AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
#define AKAZE_KEYPOINT_DETECTION_ALGORITHM_H

#include "ICPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class AKAZEKeypointDetectionAlgorithm
 *
 * @brief
 */
class AKAZEKeypointDetectionAlgorithm final : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     */
    AKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~AKAZEKeypointDetectionAlgorithm() override;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData) override;



private:


};

}

#endif // AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
