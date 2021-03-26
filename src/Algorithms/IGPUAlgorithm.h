#ifndef INTERFACE_GPU_ALGORITHM_H
#define INTERFACE_GPU_ALGORITHM_H

#include "IAlgorithm.h"

namespace Algorithms
{

class IGPUAlgorithm : public IAlgorithm
{
public:
    explicit IGPUAlgorithm(const std::shared_ptr<Config::JsonConfig>& config) :
    IAlgorithm(config)
    {
        isGPURequired_ = true;
    };
};

}

#endif // INTERFACE_GPU_ALGORITHM_H
