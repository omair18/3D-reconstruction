#include "ProcessingQueueManager.h"
#include "JsonConfig.h"

namespace DataStructures
{

ProcessingQueueManager::ProcessingQueueManager()
{

}

std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>> ProcessingQueueManager::GetQueue(const std::string &queueName) const
{
    return std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>>();
}

void ProcessingQueueManager::AddQueue(const std::string &queueName, int maxSize)
{

}

void ProcessingQueueManager::RemoveQueue(const std::string &queueName)
{

}

}