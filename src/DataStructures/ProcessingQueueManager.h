#ifndef PROCESSING_QUEUE_MANAGER_H
#define PROCESSING_QUEUE_MANAGER_H

#include <string>
#include <memory>
#include <unordered_map>

namespace Config
{
    class JsonConfig;
}

namespace DataStructures
{

class ProcessingData;

template <typename T>
class ProcessingQueue;

class ProcessingQueueManager
{

public:
    ProcessingQueueManager();

    ~ProcessingQueueManager() = default;

    [[nodiscard]] std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>> GetQueue(const std::string & queueName) const;

    void AddQueue(const std::string &queueName, int maxSize);

    void RemoveQueue(const std::string &queueName);



private:
    std::unordered_map<std::string, std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>>> queues_;
};

}

#endif // PROCESSING_QUEUE_MANAGER_H
