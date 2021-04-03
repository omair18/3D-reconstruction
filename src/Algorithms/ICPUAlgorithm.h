/**
 * @file ICPUAlgorithm.h.
 *
 * @brief
 */

#ifndef INTERFACE_CPU_ALGORITHM_H
#define INTERFACE_CPU_ALGORITHM_H

#include "IAlgorithm.h"

// forward declaration for GPU::GpuManager
namespace GPU
{
    class GpuManager;
}

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class ICPUAlgorithm
 *
 * @brief
 */
class ICPUAlgorithm : public IAlgorithm
{

public:
    /**
     * @brief
     */
    ICPUAlgorithm() :
    IAlgorithm()
    {
        isGPURequired_ = false;
    };

    /**
     * @brief
     */
    ~ICPUAlgorithm() override = default;
};

}

#endif // INTERFACE_CPU_ALGORITHM_H
