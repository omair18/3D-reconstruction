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
     * @param gpuManager
     * @param cudaStream
     */
    CUDAAKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream);

    /**
     * @brief
     */
    ~CUDAAKAZEKeypointDetectionAlgorithm() override = default;

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
    int octaves_;

    ///
    int sublayersPerOctave_;

    ///
    std::string anisotropicDiffusionFunction_;

    ///
    float threshold_;

    ///
    void* cudaStream_;

    ///
    std::shared_ptr<GPU::GPU> currentGPU_;

};

}

#endif // CUDA_AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
