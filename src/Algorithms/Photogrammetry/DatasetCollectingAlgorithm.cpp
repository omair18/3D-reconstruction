#include "DatasetCollectingAlgorithm.h"

namespace Algorithms
{

DatasetCollectingAlgorithm::DatasetCollectingAlgorithm(const std::shared_ptr<Config::JsonConfig> &config, const std::unique_ptr<GPU::GpuManager> &gpuManager, void *cudaStream)
{

}

DatasetCollectingAlgorithm::~DatasetCollectingAlgorithm()
{

}

bool DatasetCollectingAlgorithm::Process(std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    return false;
}

void DatasetCollectingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig> &config)
{

}

}