#include "DeviceToHostTransferringAlgorithm.h"

namespace Algorithms
{

DeviceToHostTransferringAlgorithm::DeviceToHostTransferringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream)
{

}

void DeviceToHostTransferringAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

bool DeviceToHostTransferringAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}


}
