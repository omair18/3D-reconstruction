#ifndef INTERFACE_CPU_ALGORITHM_H
#define INTERFACE_CPU_ALGORITHM_H

#include "IAlgorithm.h"

namespace GPU
{
    class GpuManager;
}

namespace Algorithms
{

class ICPUAlgorithm : public IAlgorithm
{
public:
    explicit ICPUAlgorithm() :
    IAlgorithm()
    {
        isGPURequired_ = false;
    };

    ~ICPUAlgorithm() override = default;
};

}

#endif // INTERFACE_CPU_ALGORITHM_H
