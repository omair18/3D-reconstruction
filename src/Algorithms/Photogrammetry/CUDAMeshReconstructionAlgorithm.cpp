#include "CUDAMeshReconstructionAlgorithm.h"

namespace Algorithms
{

CUDAMeshReconstructionAlgorithm::CUDAMeshReconstructionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
cudaStream_(cudaStream)
{

}

bool CUDAMeshReconstructionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAMeshReconstructionAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAMeshReconstructionAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAMeshReconstructionAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{

}


}
