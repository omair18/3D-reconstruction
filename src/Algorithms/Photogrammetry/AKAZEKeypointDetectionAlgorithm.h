/**
 * @file AKAZEKeypointDetectionAlgorithm.h.
 *
 * @brief
 */

#ifndef AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
#define AKAZE_KEYPOINT_DETECTION_ALGORITHM_H

#include <opencv2/core/mat.hpp>

#include "IGPUAlgorithm.h"

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
class AKAZEKeypointDetectionAlgorithm : public IGPUAlgorithm
{
    /**
     * @struct TEvolution
     *
     * @brief AKAZE nonlinear diffusion filtering evolution
     */
    struct TEvolution
    {
        /// First order spatial derivatives
        cv::Mat Lx_, Ly_;

        /// Second order spatial derivatives
        cv::Mat Lxx_, Lxy_, Lyy_;

        /// Diffusivity image
        cv::Mat Lflow_;

        /// Evolution image
        cv::Mat Lt_;

        /// Smoothed image
        cv::Mat Lsmooth_;

        /// Evolution step update
        cv::Mat Lstep_;

        /// Detector response
        cv::Mat Ldetector_;

        /// Evolution time
        float evolutionTime_ = 0;

        /// Evolution sigma. For linear diffusion t = sigma^2 / 2
        float evolutionSigma_ = 0;

        /// Image octave
        size_t octave_ = 0;

        /// Image sublevel in each octave
        size_t sublevel_ = 0;

        /// Integer sigma. For computing the feature detector responses
        size_t sigmaSize_ = 0;
    };

    /**
     * @enum DIFFUSIVITY_TYPE
     *
     * @brief
     */
    enum DIFFUSIVITY_TYPE
    {
        ///
        PM_G1 = 0,

        ///
        PM_G2 = 1,

        ///
        WEICKERT = 2,

        ///
        CHARBONNIER = 3
    };

    /**
     * @enum DESCRIPTOR_TYPE
     *
     * @brief
     */
    enum DESCRIPTOR_TYPE
    {
        ///
        MLDB = 5
    };

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
    ~AKAZEKeypointDetectionAlgorithm();

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData) override;



private:

    /// Initial octave level (-1 means that the size of the input image is duplicated)
    int octaveMin_;

    /// Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    int octaveMax_;

    /// Default number of sublevels per scale level
    int sublevels_;

    /// Width of the input image
    int imageWidth_;

    /// Height of the input image
    int imageHeight_;

    /// Base scale offset (sigma units)
    float scaleOffset_;

    /// Factor for the multiscale derivatives
    float derivativeFactor_;

    /// Smoothing factor for the derivatives
    float smoothingDerivatives_;

    /// Diffusivity type
    DIFFUSIVITY_TYPE diffusivity_;

    /// Detector response threshold to accept point
    float threshold_;

    /// Minimum detector threshold to accept a point
    float min_threshold_;

    /// Type of descriptor
    DESCRIPTOR_TYPE descriptor_;

    /// Size of the descriptor in bits. 0->Full size
    int descriptor_size_;

    /// Number of channels in the descriptor (1, 2, 3)
    int descriptor_channels_;

    /// Actual patch size is 2 * pattern_size*point.scale
    int descriptor_pattern_size_;

    /// The contrast factor parameter
    float contrast_;

    /// Percentile level for the contrast factor
    float kcontrast_percentile_;

    /// Number of bins for the contrast factor histogram
    size_t kcontrast_nbins_;

    /// Set to true for saving the scale space images
    bool save_scale_space_;

    /// Set to true for saving the detected keypoints and descriptors
    bool save_keypoints_;

    /// Number of CUDA images allocated per octave
    int ncudaimages_;

    /// Maximum number of keypoints allocated
    int maxkeypoints_;

    /// FED parameters

    /// Number of cycles
    int cycles_;

    /// Flag for reordering time steps
    bool reordering_;

    /// Vector of FED dynamic time steps
    std::vector<std::vector<float > > timeSteps_;

    /// Vector of number of steps per cycle
    std::vector<int> steps_;
};

}

#endif // AKAZE_KEYPOINT_DETECTION_ALGORITHM_H
