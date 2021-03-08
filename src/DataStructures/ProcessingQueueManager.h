#ifndef PROCESSING_QUEUE_MANAGER_H
#define PROCESSING_QUEUE_MANAGER_H

#include <string>
#include <memory>
#include <unordered_map>

namespace Config
{
    class JsonConfig;
}

class ProcessingQueueManager
{
public:
    explicit ProcessingQueueManager(const std::shared_ptr<Config::JsonConfig> & config);

    [[nodiscard]] PDQueuePtr GetQueue(const std::string & queueName) const;

    void AddQueue(const std::string &queueName, int maxSize);

    void RemoveQueue(const std::string &queueName);

    ~ProcessingQueueManager() = default;

private:
    std::unordered_map<std::string, PDQueuePtr> m_queues;
};


#endif // PROCESSING_QUEUE_MANAGER_H
