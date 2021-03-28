#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include "ProcessorManager.h"
#include "CpuProcessor.h"
#include "GpuProcessor.h"
#include "AlgorithmFactory.h"
#include "JsonConfigManager.h"
#include "ConfigNodes.h"
#include "JsonConfig.h"
#include "Logger.h"

namespace Processing
{

ProcessorManager::ProcessorManager() :
algorithmFactory_(std::make_unique<Algorithms::AlgorithmFactory>())
{

}

ProcessorManager::~ProcessorManager()
{

}

std::shared_ptr<IProcessor> Processing::ProcessorManager::GetProcessor(const std::string &processorName) const
{
    return processors_.at(processorName);
}

void ProcessorManager::Initialize(const std::shared_ptr<Config::JsonConfig>& serviceConfig,
                                  const std::unique_ptr<Config::JsonConfigManager>& configManager,
                                  const std::unique_ptr<GPU::GpuManager>& gpuManager,
                                  const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager)
{
    auto pipelineConfig = (*serviceConfig)[Config::ConfigNodes::ServiceConfig::Pipeline];

    auto processorsConfigs = pipelineConfig->GetObjects();

    for(auto& processorConfig : processorsConfigs)
    {
        ValidateProcessorConfig(processorConfig);
        AddProcessor(processorConfig, configManager, gpuManager, queueManager);
    }

}

void ProcessorManager::AddProcessor(const std::shared_ptr<Config::JsonConfig>& config,
                                    const std::unique_ptr<Config::JsonConfigManager>& configManager,
                                    const std::unique_ptr<GPU::GpuManager>& gpuManager,
                                    const std::unique_ptr<DataStructures::ProcessingQueueManager> &queueManager)
{
    std::shared_ptr<IProcessor> processor;
    auto type = (*config)[Config::ConfigNodes::ServiceConfig::PipelineConfig::Type]->ToString();
    boost::algorithm::trim(type);
    boost::algorithm::to_upper(type);

    if (type == Config::ConfigNodes::ServiceConfig::PipelineConfig::Cpu)
    {
        processor = std::make_shared<CpuProcessor>(config, queueManager);
    }
    else
    {
        processor = std::make_shared<GpuProcessor>(config, queueManager);
    }

    processor->Initialize();
    processor->InitializeAlgorithms(algorithmFactory_, configManager, gpuManager);

    processors_.insert(std::make_pair(processor->GetName(), processor));

}

void ProcessorManager::RemoveProcessor(const std::string& processorName)
{
    if (auto it = processors_.find(processorName); it == processors_.end())
    {
        LOG_WARNING() << "Failed to remove processor with name " << processorName << ". There is processor with such name.";
        return;
    }

    processors_.erase(processorName);
}

void ProcessorManager::StartProcessor(const std::string& processorName)
{
    if (auto it = processors_.find(processorName); it == processors_.end())
    {
        LOG_WARNING() << "Failed to start processor with name " << processorName << ". There is processor with such name.";
        return;
    }

    processors_.at(processorName)->Process();
}

void ProcessorManager::StopProcessor(const std::string& processorName)
{
    if (auto it = processors_.find(processorName); it == processors_.end())
    {
        LOG_WARNING() << "Failed to stop processor with name " << processorName << ". There is processor with such name.";
        return;
    }

    processors_.at(processorName)->Stop();
}

void ProcessorManager::StartAllProcessors()
{
    for(auto& processor : processors_)
    {
        processor.second->Process();
    }
}

void ProcessorManager::StopAllProcessors()
{
    for(auto& processor : processors_)
    {
        processor.second->Stop();
    }
}

void ProcessorManager::ValidateProcessorConfig(const std::shared_ptr<Config::JsonConfig>& processorConfig)
{
    if(!processorConfig->Contains(Config::ConfigNodes::ServiceConfig::PipelineConfig::Type))
    {
        LOG_ERROR() << "Invalid processor config. There is no node "
        << Config::ConfigNodes::ServiceConfig::PipelineConfig::Type << " in processor configuration.";
        throw std::runtime_error("Invalid processor config.");
    }
    else
    {
        auto type = (*processorConfig)[Config::ConfigNodes::ServiceConfig::PipelineConfig::Type]->ToString();
        boost::algorithm::trim(type);
        boost::algorithm::to_upper(type);
        if (type != Config::ConfigNodes::ServiceConfig::PipelineConfig::Cpu &&
            type != Config::ConfigNodes::ServiceConfig::PipelineConfig::Gpu)
        {
            LOG_ERROR() << "Invalid processor config. Unknown processor type " << type << ". "
            << "Processor type must be either " << Config::ConfigNodes::ServiceConfig::PipelineConfig::Cpu << " or "
            << Config::ConfigNodes::ServiceConfig::PipelineConfig::Gpu << ".";
            throw std::runtime_error("Invalid processor config.");
        }
    }

    if(!processorConfig->Contains(Config::ConfigNodes::ServiceConfig::PipelineConfig::Name))
    {
        LOG_ERROR() << "Invalid processor config. There is no node "
        << Config::ConfigNodes::ServiceConfig::PipelineConfig::Name << " in processor configuration.";
        throw std::runtime_error("Invalid processor config.");
    }

    if(!processorConfig->Contains(Config::ConfigNodes::ServiceConfig::PipelineConfig::Input) &&
       !processorConfig->Contains(Config::ConfigNodes::ServiceConfig::PipelineConfig::Output))
    {
        LOG_ERROR() << "Invalid processor config. There must be at least one of either "
        << Config::ConfigNodes::ServiceConfig::PipelineConfig::Input << " node or "
        << Config::ConfigNodes::ServiceConfig::PipelineConfig::Output << " node.";
        throw std::runtime_error("Invalid processor config.");
    }
}

}

