#include "CUDAAKAZEKeypointDetectionAlgorithm.h"

namespace Algorithms
{

CUDAAKAZEKeypointDetectionAlgorithm::CUDAAKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig> &config, const std::unique_ptr<GPU::GpuManager> &gpuManager, void *cudaStream) :
IGPUAlgorithm()
{

}

CUDAAKAZEKeypointDetectionAlgorithm::~CUDAAKAZEKeypointDetectionAlgorithm()
{

}

bool CUDAAKAZEKeypointDetectionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    return false;
}

}
