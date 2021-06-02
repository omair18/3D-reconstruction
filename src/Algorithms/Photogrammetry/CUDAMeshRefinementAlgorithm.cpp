#include "CUDAMeshRefinementAlgorithm.h"

namespace Algorithms
{

CUDAMeshRefinementAlgorithm::CUDAMeshRefinementAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
cudaStream_(cudaStream)
{

}

bool CUDAMeshRefinementAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAMeshRefinementAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAMeshRefinementAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAMeshRefinementAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{

}

}
