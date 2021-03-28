#include "AKAZEKeypointDetectionAlgorithm.h"

namespace Algorithms
{

AKAZEKeypointDetectionAlgorithm::AKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig> &config) :
IGPUAlgorithm(config)
{

}

AKAZEKeypointDetectionAlgorithm::~AKAZEKeypointDetectionAlgorithm()
{

}

bool AKAZEKeypointDetectionAlgorithm::Process(std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    return false;
}

}
