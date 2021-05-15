/**
 * @file ProcessingQueueManager.h.
 *
 * @brief Declares the class for managing processing queues.
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
 * @brief Namespace of libdatastructures library.
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
 * @brief This class is used for managing processing queues.
 */
class ProcessingQueueManager final
{

public:

    /**
     * @brief Default constructor.
     */
    ProcessingQueueManager() = default;

    /**
     * @brief Default destructor.
     */
    ~ProcessingQueueManager() = default;

    /**
     * @brief Initializes queue manager with queues listed in special node in service configuration. Reads the node of
     * service configuration containing queues configuration, validates configuration of each queue and adds a new
     * queue with configured parameters to queues_-member map.
     *
     * @param serviceConfig - Service configuration
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    /**
     * @brief Provides access to a queue with queueName-param name. If there is no such queue - returns nullptr and
     * puts a record in a log file with ERROR severity.
     *
     * @param queueName - Name of the queue for providing access
     * @return - Pointer to a queue with matching name, or nullptr in case of missing queue with matching name.
     */
    [[nodiscard]] std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>> GetQueue(const std::string& queueName) const;

    /**
     * @brief Adds a queue with name queueName-param and maximal capacity maxSize-param to queues_-member.
     * If there was already added queue with such name, recreates this queue and creates a record in a log file with
     * WARNING severity.
     *
     * @param queueName - Name of the created queue
     * @param maxSize - Maximal capacity of the created queue
     */
    void AddQueue(const std::string& queueName, int maxSize);

    /**
     * @brief Removes queue with name queueName-param from queues_-member. If there was no queue with such name,
     * does nothing and generated a record in a log file with ERROR severity.
     *
     * @param queueName - Name of the queue to be removed
     */
    void RemoveQueue(const std::string& queueName);

private:

    /**
     * @brief Checks weather queueConfig-param JSON config has all required fields.
     *
     * @param queueConfig - Pointer to a JSON config to validate
     */
    static void ValidateQueueConfig(const std::shared_ptr<Config::JsonConfig>& queueConfig);

    /// Unordered map, where key is the name of the queue, value - shared pointer to the queue.
    std::unordered_map<std::string, std::shared_ptr<ProcessingQueue<std::shared_ptr<ProcessingData>>>> queues_;
};

}

#endif // PROCESSING_QUEUE_MANAGER_H
