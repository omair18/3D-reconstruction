#include "CUDAPointCloudDensificationAlgorithm.h"

namespace Algorithms
{

CUDAPointCloudDensificationAlgorithm::CUDAPointCloudDensificationAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
cudaStream_(cudaStream)
{

}

bool CUDAPointCloudDensificationAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAPointCloudDensificationAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAPointCloudDensificationAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAPointCloudDensificationAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{

}

}