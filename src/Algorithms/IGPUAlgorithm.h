/**
 * @file IGPUAlgorithm.h.
 *
 * @brief
 */

#ifndef INTERFACE_GPU_ALGORITHM_H
#define INTERFACE_GPU_ALGORITHM_H

#include "IAlgorithm.h"

// forward declaration for GPU::GpuManager and GPU::GPU
namespace GPU
{
    class GpuManager;
    struct GPU;
}

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class IGPUAlgorithm
 *
 * @brief
 */
class IGPUAlgorithm : public IAlgorithm
{

public:

    /**
     * @brief
     */
    IGPUAlgorithm() :
    IAlgorithm()
    {
        isGPURequired_ = true;
    };

    /**
     * @brief
     */
    ~IGPUAlgorithm() override = default;
};

}

#endif // INTERFACE_GPU_ALGORITHM_H
