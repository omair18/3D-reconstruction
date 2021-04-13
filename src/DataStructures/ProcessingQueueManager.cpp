#include "ProcessingQueueManager.h"
#include "ProcessingQueue.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace DataStructures
{

std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>> ProcessingQueueManager::GetQueue(const std::string &queueName) const
{
    if (auto it = queues_.find(queueName); it != queues_.end())
    {
        return it->second;
    }

    LOG_ERROR() << "Failed to get queue with name << " << queueName <<". There is queue with such name.";

    return nullptr;

}

void ProcessingQueueManager::AddQueue(const std::string &queueName, int maxSize)
{
    if (auto it = queues_.find(queueName); it != queues_.end())
    {
        LOG_WARNING() << "Recreating the queue with name " << queueName << ".";
    }

    queues_.insert({queueName, std::make_shared<ProcessingQueue<std::shared_ptr<ProcessingData>>>(queueName, maxSize)});

    LOG_TRACE() << "A queue with name " << queueName << " and max size " << maxSize << " was added.";

}

void ProcessingQueueManager::RemoveQueue(const std::string &queueName)
{
    if (auto it = queues_.find(queueName); it == queues_.end())
    {
        LOG_WARNING() << "Failed to remove queue with name " << queueName << ". There is queue with such name.";

        return;
    }

    queues_.erase(queueName);
}

void ProcessingQueueManager::Initialize(const std::shared_ptr<Config::JsonConfig> &serviceConfig)
{
    auto queuesConfigsArray = (*serviceConfig)[Config::ConfigNodes::ServiceConfig::Queues];

    auto queuesConfigs = queuesConfigsArray->GetObjects();
    for(auto& queueConfig : queuesConfigs)
    {
        auto queueName = (*queueConfig)[Config::ConfigNodes::ServiceConfig::QueueConfig::Name]->ToString();
        auto queueSize = (*queueConfig)[Config::ConfigNodes::ServiceConfig::QueueConfig::Size]->ToInt32();
        AddQueue(queueName, queueSize);
    }
}

}