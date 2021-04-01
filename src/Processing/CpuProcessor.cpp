#include "ICPUAlgorithm.h"
#include "CpuProcessor.h"
#include "ProcessingData.h"
#include "ProcessingQueue.h"
#include "JsonConfig.h"
#include "JsonConfigManager.h"
#include "ConfigNodes.h"
#include "IAlgorithmFactory.h"
#include "Logger.h"

namespace Processing
{

CpuProcessor::CpuProcessor(const std::shared_ptr<Config::JsonConfig> &config, const std::unique_ptr<DataStructures::ProcessingQueueManager> &queueManager) :
IProcessor(config, queueManager)
{

}

CpuProcessor::~CpuProcessor()
{
    thread_.Destroy();
}

void CpuProcessor::Process()
{
    LOG_TRACE() << "Starting CPU processor " << name_ << " ...";
    thread_.Start();
}

void CpuProcessor::Initialize()
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
            LOG_ERROR() << "Received exception on CPU processor " << name_ <<". " << exception.what();
        }
    });
}

void Processing::CpuProcessor::InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
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
                auto algorithm = algorithmFactory->Create(algorithmConfig, gpuManager, nullptr);
                if(algorithm->RequiresGPU())
                {
                    LOG_ERROR() << "Invalid processor's algorithms configuration. Cannot create GPU algorithm on " \
                    "CPU processor";
                    throw std::runtime_error("Invalid processor's algorithms configuration");
                }
                processingAlgorithms_.push_back(std::move(algorithm));
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

void Processing::CpuProcessor::Stop()
{
    thread_.Stop();
}

bool CpuProcessor::IsStarted()
{
    return thread_.IsStarted();
}

void CpuProcessor::ValidateAlgorithmConfig(const std::shared_ptr<Config::JsonConfig> &algorithmConfig)
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