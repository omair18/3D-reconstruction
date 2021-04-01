#ifndef INTERFACE_GPU_ALGORITHM_H
#define INTERFACE_GPU_ALGORITHM_H

#include "IAlgorithm.h"

namespace GPU
{
    class GpuManager;
    struct GPU;
}

namespace Algorithms
{

class IGPUAlgorithm : public IAlgorithm
{
public:
    explicit IGPUAlgorithm() :
    IAlgorithm()
    {
        isGPURequired_ = true;
    };

    ~IGPUAlgorithm() override = default;
};

}

#endif // INTERFACE_GPU_ALGORITHM_H
