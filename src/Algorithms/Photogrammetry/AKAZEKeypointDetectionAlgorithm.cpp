#include "AKAZEKeypointDetectionAlgorithm.h"

namespace Algorithms
{

AKAZEKeypointDetectionAlgorithm::AKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig> &config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager> &gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm()
{

}

AKAZEKeypointDetectionAlgorithm::~AKAZEKeypointDetectionAlgorithm()
{

}

bool AKAZEKeypointDetectionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

}
