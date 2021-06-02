#include "CUDAMeshTexturingAlgorithm.h"

namespace Algorithms
{

CUDAMeshTexturingAlgorithm::CUDAMeshTexturingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
cudaStream_(cudaStream)
{

}

bool CUDAMeshTexturingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAMeshTexturingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAMeshTexturingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAMeshTexturingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{

}

}