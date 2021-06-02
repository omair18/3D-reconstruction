#include "CUDAKeyPointFilteringAlgorithm.h"

namespace Algorithms
{

CUDAKeyPointFilteringAlgorithm::CUDAKeyPointFilteringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
cudaStream_(cudaStream)
{

}

bool CUDAKeyPointFilteringAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAKeyPointFilteringAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAKeyPointFilteringAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAKeyPointFilteringAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{

}


}
