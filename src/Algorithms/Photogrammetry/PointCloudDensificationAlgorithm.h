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

private:

    /// How many times to scale down the images before point cloud computation.
    unsigned int resolutionLevel_ = 1;

    /// Do not scale images higher than this resolution.
    unsigned int maxResolution_ = 3200;

    /// Do not scale images lower than this resolution.
    unsigned int minResolution_ = 640;

    /// Minimal number of agreeing views to validate a depth.
    unsigned int minViews_ = 2;

    /// Maximal number of neighbor images used to compute the depth-map for the reference image.
    unsigned int maxViews_ = 12;

    /// Minimal number of images that agrees with an estimate during fusion in order to consider it inlier.
    unsigned int minViewsFuse_ = 3;

    /// Minimal number of images that agrees with an estimate in order to consider it inlier.
    unsigned int minViewsFilter_ = 2;

    /// Minimal number of images that agrees with an estimate in order to consider it inlier (0 - disabled).
    unsigned int minViewsFilterAdjust_ = 1;

    /// Minimal number of views so that the point is considered for approximating the depth-maps (<2 - random initialization).
    unsigned int minViewsTrustPoint_ = 2;

    /// Number of views used for depth-map estimation (0 - all neighbor views available).
    unsigned int numViews_ = 5;

    /// Adjust depth estimates during filtering.
    bool filterAdjust_ = true;

    /// Add support points at image corners with nearest neighbor disparities.
    bool addCorners_ = true;

    /// Minimal score to consider a neighbor images (0 - disabled).
    float viewMinScore_ = 2.0;

    /// Minimal score ratio to consider a neighbor images.
    float viewMinScoreRatio_ = 0.3;

    /// Minimal shared area for accepting the depth triangulation.
    float minArea_ = 0.05;

    /// Minimal angle for accepting the depth triangulation.
    float minAngle_ = 3.0;

    /// Optimal angle for computing the depth triangulation.
    float optimalAngle_ = 10.0;

    /// Maximal angle for accepting the depth triangulation.
    float maxAngle_ = 65.0;

    /// Minimal texture variance accepted when matching two patches (0 - disabled).
    float descriptorMinMagnitudeThreshold_ = 0.01;

    /// Maximal variance allowed for the depths during refinement.
    float depthDiffThreshold_ = 0.01;

    /// Maximal variance allowed for the normal during fusion (degrees).
    float normalDiffThreshold_ = 25;

    /// Pairwise cost scale to match the unary cost.
    float pairwiseMul_ = 0.3;

    /// MRF optimizer stop epsilon.
    float optimizerEps_ = 0.001;

    /// MRF optimizer max number of iterations.
    int optimizerMaxIterations_ = 80;

    /// Maximal size of a speckle (small speckles get removed).
    unsigned int speckleSize_ = 100;

    /// Interpolate small gaps (left<->right, top<->bottom).
    unsigned int interpolationGapSize_ = 7;

    /// Should we filter the extracted depth-maps?
    unsigned int optimize_ = 7;

    /// Estimate the colors for the dense point-cloud (0 - disabled, 1 - final, 2 - estimate).
    unsigned int estimateColors_ = 2;

    /// Estimate the normals for the dense point-cloud (0 - disabled, 1 - final, 2 - estimate).
    unsigned int estimateNormals_ = 2;

    /// Maximum 1-NCC score accepted for a match.
    float NCCThresholdKeep = 0.55;

    /// Number of iterations for depth-map refinement.
    unsigned int estimationIterations_ = 4;

    /// Number of iterations for random assignment per pixel.
    unsigned int randomIterations_ = 6;

    /// Maximum number of iterations to skip during random assignment.
    unsigned int randomMaxScale_ = 2;

    /// Depth range ratio of the current estimate for random plane assignment.
    float randomDepthRatio_ = 0.003;

    /// Angle 1 range for random plane assignment (degrees).
    float randomAngle1Range_ = 16.0;

    /// Angle 2 range for random plane assignment (degrees).
    float randomAngle2Range_ = 10.0;

    /// Depth variance used during neighbor smoothness assignment (ratio).
    float randomSmoothDepth_ = 0.02;

    /// Normal variance used during neighbor smoothness assignment (degrees).
    float randomSmoothNormal_ = 13;

    /// Score factor used to encourage smoothness (1 - disabled).
    float randomSmoothBonus_ = 0.93;
};

}

#endif // POINT_CLOUD_DENSIFICATION_ALGORITHM_H
