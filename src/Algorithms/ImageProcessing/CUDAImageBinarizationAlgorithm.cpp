#include <nppi_color_conversion.h>

#include "CUDAImageBinarizationAlgorithm.h"
#include "GpuManager.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace Algorithms
{

CUDAImageBinarizationAlgorithm::CUDAImageBinarizationAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
cudaStream_(cudaStream),
allowUnconfiguredChannels_(false),
currentGPU_(gpuManager->GetCurrentGPU()),
binarizationBuffer_()
{
    InitializeInternal(config);
}

bool CUDAImageBinarizationAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    return false;
}

void CUDAImageBinarizationAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void CUDAImageBinarizationAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::CUDAImageBinarizationAlgorithm::AllowUnconfiguredChannels))
    {
        LOG_ERROR() << "Invalid CUDA image binarization algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::CUDAImageBinarizationAlgorithm::AllowUnconfiguredChannels << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::CUDAImageBinarizationAlgorithm::BinarizationCoefficients))
    {
        LOG_ERROR() << "Invalid CUDA image binarization algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::CUDAImageBinarizationAlgorithm::BinarizationCoefficients
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void CUDAImageBinarizationAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(binarizationCoefficients_.empty())
    {
        LOG_TRACE() << "Initializing CUDA image binarization algorithm ...";
        ValidateConfig(config);

        allowUnconfiguredChannels_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::CUDAImageBinarizationAlgorithm::AllowUnconfiguredChannels]->ToBool();

        auto coefficients = (*config)[Config::ConfigNodes::AlgorithmsConfig::CUDAImageBinarizationAlgorithm::BinarizationCoefficients];

        coefficients->GetObjects()

    }
    else
    {
        LOG_WARNING() << "CUDA image decoding algorithm is already initialized.";
    }
}

void CUDAImageBinarizationAlgorithm::AllocateBuffer(int width, int height)
{

}

}