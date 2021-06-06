/**
 * @file MeshReconstructionAlgorithm.h.
 *
 * @brief
 */

#ifndef MESH_RECONSTRUCTION_ALGORITHM_H
#define MESH_RECONSTRUCTION_ALGORITHM_H

#include "ICPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class MeshReconstructionAlgorithm
 *
 * @brief
 */
class MeshReconstructionAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    MeshReconstructionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~MeshReconstructionAlgorithm() override = default;

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

    /// Minimal distance in pixels between the projection of two 3D points to consider them different while triangulating (0 - disabled).
    float distanceInsert_;

    /// Considers all view weights 1 instead of the available weight.
    bool useConstantWeight_;

    /// Exploits the free-space support in order to reconstruct weakly-represented surfaces.
    bool useFreeSpaceSupport_;

    /// Multiplier adjusting the minimum thickness considered during visibility weighting.
    float thicknessFactor_;

    /// Multiplier adjusting the quality weight considered during graph-cut.
    float qualityFactor_;

    /// Decimation factor in range (0..1] to be applied to the reconstructed surface (1 - disabled).
    float decimateMesh_;

    /// Spurious factor for removing faces with too long edges or isolated components (0 - disabled).
    float removeSpurious_;

    /// Flag controlling the removal of spike faces.
    bool removeSpikes_;

    /// Try to close small holes in the reconstructed surface (0 - disabled).
    unsigned int closeHoles_;

    /// Number of iterations to smooth the reconstructed surface (0 - disabled).
    unsigned int smoothMesh_;

};

}

#endif // MESH_RECONSTRUCTION_ALGORITHM_H
