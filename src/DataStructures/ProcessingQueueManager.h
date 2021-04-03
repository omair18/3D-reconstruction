/**
 * @file ProcessingQueueManager.h.
 *
 * @brief
 */

#ifndef PROCESSING_QUEUE_MANAGER_H
#define PROCESSING_QUEUE_MANAGER_H

#include <string>
#include <memory>
#include <unordered_map>

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

/**
 * @namespace DataStructures
 *
 * @brief
 */
namespace DataStructures
{

// forward declaration for DataStructures::ProcessingData
class ProcessingData;

// forward declaration for DataStructures::ProcessingQueue
template <typename T>
class ProcessingQueue;

/**
 * @class ProcessingQueueManager
 *
 * @brief
 */
class ProcessingQueueManager final
{

public:

    /**
     * @brief
     */
    ProcessingQueueManager() = default;

    /**
     * @brief
     */
    ~ProcessingQueueManager() = default;

    /**
     * @brief
     *
     * @param serviceConfig
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    /**
     * @brief
     *
     * @param queueName
     * @return
     */
    [[nodiscard]] std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>> GetQueue(const std::string & queueName) const;

    /**
     * @brief
     *
     * @param queueName
     * @param maxSize
     */
    void AddQueue(const std::string &queueName, int maxSize);

    /**
     * @brief
     *
     * @param queueName
     */
    void RemoveQueue(const std::string &queueName);

private:

    ///
    std::unordered_map<std::string, std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>>> queues_;
};

}

#endif // PROCESSING_QUEUE_MANAGER_H
