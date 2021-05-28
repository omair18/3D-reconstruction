#include "HostToDeviceTransferringAlgorithm.h"

namespace Algorithms
{

HostToDeviceTransferringAlgorithm::HostToDeviceTransferringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream)
{

}

bool HostToDeviceTransferringAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void HostToDeviceTransferringAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

}