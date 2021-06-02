#include "CUDAKeyPointMatchingAlgorithm.h"

namespace Algorithms
{

CUDAKeyPointMatchingAlgorithm::CUDAKeyPointMatchingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
cudaStream_(cudaStream)
{

}

bool CUDAKeyPointMatchingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAKeyPointMatchingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAKeyPointMatchingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAKeyPointMatchingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{

}

}