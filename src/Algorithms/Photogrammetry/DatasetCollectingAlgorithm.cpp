#include "DatasetCollectingAlgorithm.h"
#include "Logger.h"
#include "ConfigNodes.h"
#include "ProcessingData.h"

namespace Algorithms
{

DatasetCollectingAlgorithm::DatasetCollectingAlgorithm(const std::shared_ptr<Config::JsonConfig> &config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager> &gpuManager, [[maybe_unused]] void *cudaStream) :
expireTimeoutSeconds_(120)
{

}

bool DatasetCollectingAlgorithm::Process(std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    if()
    {

    }
    return false;
}

void DatasetCollectingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig> &config)
{
    LOG_TRACE() << "Initializing " << Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::DatasetCollectingAlgorithm
    << " ...";

}

void DatasetCollectingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig> &config)
{

}

void DatasetCollectingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig> &config)
{

}

}