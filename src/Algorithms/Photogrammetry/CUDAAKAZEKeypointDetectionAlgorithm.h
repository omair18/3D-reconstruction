/**
 * @file CUDAAKAZEKeypointDetectionAlgorithm.h.
 *
 * @brief
 */

#ifndef CUDA_AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
#define CUDA_AKAZE_KEYPOINT_DETECTION_ALGORITHM_H

#include "IGPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class CUDAAKAZEKeypointDetectionAlgorithm
 *
 * @brief
 */
class CUDAAKAZEKeypointDetectionAlgorithm final : public IGPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     */
    CUDAAKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~CUDAAKAZEKeypointDetectionAlgorithm() override;

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

#endif // CUDA_AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
