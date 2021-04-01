#include "CUDAImageDecodingAlgorithm.h"

Algorithms::CUDAImageDecodingAlgorithm::CUDAImageDecodingAlgorithm(const std::shared_ptr<Config::JsonConfig> &config,
                                                                   const std::unique_ptr<GPU::GpuManager> &gpuManager,
                                                                   void *cudaStream)
{

}

Algorithms::CUDAImageDecodingAlgorithm::~CUDAImageDecodingAlgorithm()
{

}

bool Algorithms::CUDAImageDecodingAlgorithm::Process(std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    return false;
}

void Algorithms::CUDAImageDecodingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig> &config)
{

}
