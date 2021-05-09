#include "AKAZEKeypointDetectionAlgorithm.h"

namespace Algorithms
{

AKAZEKeypointDetectionAlgorithm::AKAZEKeypointDetectionAlgorithm(
        const std::shared_ptr<Config::JsonConfig> &config,
        const std::unique_ptr<GPU::GpuManager> &gpuManager,
        void *cudaStream) :
IGPUAlgorithm()
{

}

AKAZEKeypointDetectionAlgorithm::~AKAZEKeypointDetectionAlgorithm()
{

}

bool AKAZEKeypointDetectionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    return false;
}

}
