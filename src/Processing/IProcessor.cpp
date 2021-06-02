#include "IProcessor.h"
#include "IAlgorithm.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "ProcessingQueueManager.h"
#include "Logger.h"

namespace Processing
{

IProcessor::IProcessor(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager)
{
    auto name = (*config)[Config::ConfigNodes::ServiceConfig::PipelineConfig::Name]->ToString();
    std::string inputQueueName;
    std::string outputQueueName;

    SetName(std::move(name));

    if(config->Contains(Config::ConfigNodes::ServiceConfig::PipelineConfig::Input))
    {
        inputQueueName = (*config)[Config::ConfigNodes::ServiceConfig::PipelineConfig::Input]->ToString();
    }

    if(config->Contains(Config::ConfigNodes::ServiceConfig::PipelineConfig::Output))
    {
        outputQueueName = (*config)[Config::ConfigNodes::ServiceConfig::PipelineConfig::Output]->ToString();
    }

    if(!inputQueueName.empty())
    {
        auto inputQueue = queueManager->GetQueue(inputQueueName);
        if(inputQueue)
        {
            inputQueue_ = std::move(inputQueue);
        }
        else
        {
            LOG_ERROR() << "Failed to set input queue with name " << inputQueueName << ". Queue with such  name doesn't exist.";
            throw std::runtime_error("Processor initialization error");
        }
    }

    if(!outputQueueName.empty())
    {
        auto outputQueue = queueManager->GetQueue(outputQueueName);
        if(outputQueue)
        {
            outputQueue_ = std::move(outputQueue);
        }
        else
        {
            LOG_ERROR() << "Failed to set output queue with name " << outputQueueName << ". Queue with such  name doesn't exist.";
            throw std::runtime_error("Processor initialization error");
        }
    }
}

}