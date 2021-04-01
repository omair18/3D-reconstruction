#include <cuda_runtime.h>

#include "IGPUAlgorithm.h"
#include "GpuProcessor.h"
#include "ProcessingQueue.h"
#include "JsonConfigManager.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "ProcessingData.h"
#include "IAlgorithmFactory.h"
#include "Logger.h"

namespace Processing
{

GpuProcessor::GpuProcessor(const std::shared_ptr<Config::JsonConfig> &config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager) :
IProcessor(config, queueManager)
{
    cudaError_t status;
    status = cudaStreamCreate(&cudaStream_);
    if(status != cudaError::cudaSuccess)
    {
        LOG_ERROR() << "Failed to create CUDA stream for GPU processor " << name_ <<". Details: "
                    << status << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
        throw std::runtime_error("CUDA runtime error.");
    }
}

void GpuProcessor::Process()
{
    LOG_TRACE() << "Starting GPU processor " << name_ << " ...";
    thread_.Start();
}

GpuProcessor::~GpuProcessor()
{
    thread_.Destroy();
    while (thread_.IsStarted())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if(cudaStream_)
    {
        cudaStreamDestroy(cudaStream_);
    }
}

void GpuProcessor::Stop()
{
    thread_.Stop();
}

void GpuProcessor::InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                                        const std::unique_ptr<Config::JsonConfigManager>& configManager,
                                        const std::unique_ptr<GPU::GpuManager>& gpuManager)
{
    auto processorAlgorithmsConfig = configManager->GetConfig(name_);

    if(processorAlgorithmsConfig)
    {
        if(processorAlgorithmsConfig->Contains(Config::ConfigNodes::AlgorithmsConfig::Algorithms))
        {
            auto algorithmsConfigArray = (*processorAlgorithmsConfig)[Config::ConfigNodes::AlgorithmsConfig::Algorithms]->GetObjects();
            for(auto& algorithmConfig : algorithmsConfigArray)
            {
                ValidateAlgorithmConfig(algorithmConfig);
                processingAlgorithms_.push_back(std::move(algorithmFactory->Create(algorithmConfig, gpuManager, cudaStream_)));
            }
        }
        else
        {
            LOG_ERROR() << "";
            throw std::runtime_error("");
        }
    }
    else
    {
        LOG_ERROR() << "";
        throw std::runtime_error("Processor algorithms configuration is missing.");
    }
}

void GpuProcessor::Initialize()
{
    thread_.SetExecutableFunction([&]()
    {
        bool wasProcessingSuccessful = true;
        try
        {
            std::shared_ptr<DataStructures::ProcessingData> processingData;
            if (inputQueue_)
            {
                inputQueue_->Get(processingData);
            }
            else
            {
                processingData = std::make_shared<DataStructures::ProcessingData>();
            }

            for (auto& algorithm : processingAlgorithms_)
            {
                if(!algorithm->Process(processingData))
                {
                    wasProcessingSuccessful = false;
                    break;
                }
            }

            if(outputQueue_ && wasProcessingSuccessful)
            {
                outputQueue_->Put(processingData);
            }


        }
        catch (std::exception& exception)
        {
            LOG_ERROR() << "Received exception on GPU processor " << name_ <<". " << exception.what();
        }
    });
}

bool GpuProcessor::IsStarted()
{
    return thread_.IsStarted();
}

void GpuProcessor::ValidateAlgorithmConfig(const std::shared_ptr<Config::JsonConfig> &algorithmConfig)
{
    if(!algorithmConfig->Contains(Config::ConfigNodes::AlgorithmsConfig::Name))
    {
        LOG_ERROR() << "";
        throw std::runtime_error("");
    }

    if(!algorithmConfig->Contains(Config::ConfigNodes::AlgorithmsConfig::Configuration))
    {
        LOG_ERROR() << "";
        throw std::runtime_error("");
    }
}

}


