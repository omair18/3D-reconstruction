/**
 * @file MeshTexturingAlgorithm.h.
 *
 * @brief
 */

#ifndef MESH_TEXTURING_ALGORITHM_H
#define MESH_TEXTURING_ALGORITHM_H

#include "ICPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class MeshTexturingAlgorithm
 *
 * @brief
 */
class MeshTexturingAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    MeshTexturingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~MeshTexturingAlgorithm() override = default;

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

    /// How many times to scale down the images before mesh refinement.
    unsigned int resolutionLevel_ = 0;

    /// Do not scale images lower than this resolution.
    unsigned int minResolution_ = 640;

    /// Threshold used to find and remove outlier face textures (0 - disabled).
    float outlierThreshold_ = 6e-2f;

    /// Ratio used to adjust the preference for more compact patches (1 - best quality/worst compactness, ~0 - worst quality/best compactness).
    float ratioDataSmoothness_ = 0.1f;

    /// Generate uniform texture patches using global seam leveling.
    bool globalSeamLeveling_ = true;

    /// Generate uniform texture patch borders using local seam leveling.
    bool localSeamLeveling_ = true;

    /// Texture size should be a multiple of this value (0 - power of two).
    unsigned int textureSizeMultiple_ = 0;

    /// Specify the heuristic used when deciding where to place a new patch (0 - best fit, 3 - good speed, 100 - best speed).
    unsigned int rectPackingHeuristic_ = 0;

    /// Color used for faces not covered by any image.
    unsigned int colorEmpty_ = 16744231;

};

}

#endif // MESH_TEXTURING_ALGORITHM_H
