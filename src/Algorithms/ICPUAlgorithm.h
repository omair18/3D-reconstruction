#ifndef INTERFACE_CPU_ALGORITHM_H
#define INTERFACE_CPU_ALGORITHM_H

#include "IAlgorithm.h"

namespace Algorithms
{

class ICPUAlgorithm : public IAlgorithm
{
public:
    explicit ICPUAlgorithm(const std::shared_ptr<Config::JsonConfig>& config) :
    IAlgorithm(config)
    {
        isGPURequired_ = false;
    };
};

}

#endif // INTERFACE_CPU_ALGORITHM_H
