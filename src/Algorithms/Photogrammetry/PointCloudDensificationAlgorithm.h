/**
 * @file PointCloudDensificationAlgorithm.h.
 *
 * @brief
 */

#ifndef POINT_CLOUD_DENSIFICATION_ALGORITHM_H
#define POINT_CLOUD_DENSIFICATION_ALGORITHM_H

#include "ICPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class PointCloudDensificationAlgorithm
 *
 * @brief
 */
class PointCloudDensificationAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    PointCloudDensificationAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~PointCloudDensificationAlgorithm() override = default;

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
    bool isInitialized_;

    ///
    bool saveResult_;

    /// How many times to scale down the images before point cloud computation.
    unsigned int resolutionLevel_;

    /// Do not scale images higher than this resolution.
    unsigned int maxResolution_;

    /// Do not scale images lower than this resolution.
    unsigned int minResolution_;

    /// Minimal number of agreeing views to validate a depth.
    unsigned int minViews_;

    /// Maximal number of neighbor images used to compute the depth-map for the reference image.
    unsigned int maxViews_;

    /// Minimal number of images that agrees with an estimate during fusion in order to consider it inlier.
    unsigned int minViewsFuse_;

    /// Minimal number of images that agrees with an estimate in order to consider it inlier.
    unsigned int minViewsFilter_;

    /// Minimal number of images that agrees with an estimate in order to consider it inlier (0 - disabled).
    unsigned int minViewsFilterAdjust_;

    /// Minimal number of views so that the point is considered for approximating the depth-maps (<2 - random initialization).
    unsigned int minViewsTrustPoint_;

    /// Number of views used for depth-map estimation (0 - all neighbor views available).
    unsigned int numberOfViews_;

    /// Adjust depth estimates during filtering.
    bool filterAdjust_;

    /// Add support points at image corners with nearest neighbor disparities.
    bool addCorners_;

    /// Minimal score to consider a neighbor images (0 - disabled).
    float viewMinScore_;

    /// Minimal score ratio to consider a neighbor images.
    float viewMinScoreRatio_;

    /// Minimal shared area for accepting the depth triangulation.
    float minArea_;

    /// Minimal angle for accepting the depth triangulation.
    float minAngle_;

    /// Optimal angle for computing the depth triangulation.
    float optimalAngle_;

    /// Maximal angle for accepting the depth triangulation.
    float maxAngle_;

    /// Minimal texture variance accepted when matching two patches (0 - disabled).
    float descriptorMinMagnitudeThreshold_;

    /// Maximal variance allowed for the depths during refinement.
    float depthDiffThreshold_;

    /// Maximal variance allowed for the normal during fusion (degrees).
    float normalDiffThreshold_;

    /// Pairwise cost scale to match the unary cost.
    float pairwiseMul_;

    /// MRF optimizer stop epsilon.
    float optimizerEps_;

    /// MRF optimizer max number of iterations.
    int optimizerMaxIterations_;

    /// Maximal size of a speckle (small speckles get removed).
    unsigned int speckleSize_;

    /// Interpolate small gaps (left<->right, top<->bottom).
    unsigned int interpolationGapSize_;

    /// Should we filter the extracted depth-maps?
    unsigned int optimize_;

    /// Estimate the colors for the dense point-cloud (0 - disabled, 1 - final, 2 - estimate).
    unsigned int estimateColors_;

    /// Estimate the normals for the dense point-cloud (0 - disabled, 1 - final, 2 - estimate).
    unsigned int estimateNormals_;

    /// Maximum 1-NCC score accepted for a match.
    float NCCThresholdKeep_;

    /// Number of iterations for depth-map refinement.
    unsigned int estimationIterations_;

    /// Number of iterations for random assignment per pixel.
    unsigned int randomIterations_;

    /// Maximum number of iterations to skip during random assignment.
    unsigned int randomMaxScale_;

    /// Depth range ratio of the current estimate for random plane assignment.
    float randomDepthRatio_;

    /// Angle 1 range for random plane assignment (degrees).
    float randomAngle1Range_;

    /// Angle 2 range for random plane assignment (degrees).
    float randomAngle2Range_;

    /// Depth variance used during neighbor smoothness assignment (ratio).
    float randomSmoothDepth_;

    /// Normal variance used during neighbor smoothness assignment (degrees).
    float randomSmoothNormal_;

    /// Score factor used to encourage smoothness (1 - disabled).
    float randomSmoothBonus_;
};

}

#endif // POINT_CLOUD_DENSIFICATION_ALGORITHM_H
