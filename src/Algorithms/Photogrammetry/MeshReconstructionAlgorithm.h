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

private:

    /// Minimal distance in pixels between the projection of two 3D points to consider them different while triangulating (0 - disabled).
    float distInsert_ = 2.5;

    /// Considers all view weights 1 instead of the available weight.
    bool useConstantWeight_ = true;

    /// Exploits the free-space support in order to reconstruct weakly-represented surfaces.
    bool useFreeSpaceSupport_ = false;

    /// Multiplier adjusting the minimum thickness considered during visibility weighting.
    float thicknessFactor = 1;

    /// Multiplier adjusting the quality weight considered during graph-cut.
    float qualityFactor_ = 2;

    /// Decimation factor in range (0..1] to be applied to the reconstructed surface (1 - disabled).
    float decimateMesh_ = 1;

    /// Spurious factor for removing faces with too long edges or isolated components (0 - disabled).
    float removeSpurious_ = 30;

    /// Flag controlling the removal of spike faces.
    bool removeSpikes_ = true;

    /// Try to close small holes in the reconstructed surface (0 - disabled).
    unsigned closeHoles_ = 30;

    /// Number of iterations to smooth the reconstructed surface (0 - disabled).
    unsigned smoothMesh_ = 2;

};

}

#endif // MESH_RECONSTRUCTION_ALGORITHM_H
