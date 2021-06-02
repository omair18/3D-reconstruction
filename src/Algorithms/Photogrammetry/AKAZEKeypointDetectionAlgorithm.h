/**
 * @file AKAZEKeypointDetectionAlgorithm.h.
 *
 * @brief
 */

#ifndef AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
#define AKAZE_KEYPOINT_DETECTION_ALGORITHM_H

#include <opencv2/core/mat.hpp>

#include "ICPUAlgorithm.h"

namespace cv
{
    class AKAZE;
}

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
     * @param gpuManager
     * @param cudaStream
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
    cv::Mat buffer_;

    ///
    std::shared_ptr<cv::AKAZE> akaze_;

};

}

#endif // AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
