#include "CUDAAKAZEKeypointDetectionAlgorithm.h"
#include "GpuManager.h"

namespace Algorithms
{

CUDAAKAZEKeypointDetectionAlgorithm::CUDAAKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
threshold_(0.f),
octaves_(0),
sublayersPerOctave_(0),
anisotropicDiffusionFunction_(""),
cudaStream_(cudaStream),
currentGPU_(gpuManager->GetCurrentGPU())
{

}

bool CUDAAKAZEKeypointDetectionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAAKAZEKeypointDetectionAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAAKAZEKeypointDetectionAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{

}

void CUDAAKAZEKeypointDetectionAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{

}

}
