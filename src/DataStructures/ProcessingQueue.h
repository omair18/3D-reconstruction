/**
 * @file ProcessingQueue.h.
 *
 * @brief
 */

#ifndef PROCESSING_QUEUE_H
#define PROCESSING_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <utility>

/**
 * @namespace DataStructures
 *
 * @brief Namespace of libdatastructures library.
 */
namespace DataStructures
{

/**
 * @class ProcessingQueue
 *
 * @brief
 *
 * @tparam Type
 */
template<class Type>
class ProcessingQueue final
{
public:

    /**
     * @brief
     *
     * @param name
     * @param maxSize
     */
    ProcessingQueue(std::string name, int maxSize) :
    name_(std::move(name)), maxSize_(maxSize)
    {

    };

    /**
     * @brief
     */
    ~ProcessingQueue() = default;

    /**
     * @brief Puts an element to the queue.
     *
     * @param element - Element to put
     */
    void Put(const Type& element)
    {
        {
            std::unique_lock<std::mutex> lock(dequeMutex_);
            dequeNotFullCV_.wait(lock, [&](){
                return deque_.size() < maxSize_;
            });
            deque_.push_back(element);
        }
        dequeNotEmptyCV_.notify_one();
    }

    /**
     * @brief Get an element from the queue
     *
     * @param element
     */
    void Get(Type& element)
    {
        {
            std::unique_lock<std::mutex> lock(dequeMutex_);
            dequeNotEmptyCV_.wait(lock, [&](){
                return !deque_.empty();
            });
            element = deque_.front();
            deque_.pop_front();
            dequeNotFullCV_.notify_one();
        }
    }

    /**
     * @brief Clear the queue
     */
    void Clear()
    {
        std::lock_guard<std::mutex> lock(dequeMutex_);
        deque_.clear();
    }

    /**
     * @brief Check if the queue empty
     *
     * @return
     */
    bool Empty()
    {
        std::lock_guard<std::mutex> lock(dequeMutex_);
        return deque_.empty();
    }

    /**
     * @brief Returns the length of the queue
     *
     * @return
     */
    size_t Size()
    {
        std::lock_guard<std::mutex> lock(dequeMutex_);
        return deque_.size();
    }

    /**
     * @brief Get name of the queue
     *
     * @return Name of the queue
     */
    const std::string& GetName()
    {
        return name_;
    }

    /**
     * @brief Sets name of the queue
     *
     * @param name - new name of the queue
     */
    void SetName(const std::string& name)
    {
        name_ = name;
    }

protected:
    /// The queue.
    std::deque<Type> deque_;

    ///The mutex of the queue.
    std::mutex dequeMutex_;

    /// The conditional variable of the queue.
    std::condition_variable dequeNotEmptyCV_;

    /// The conditional variable of the queue.
    std::condition_variable dequeNotFullCV_;

    /// The name of the queue.
    std::string name_;

    /// The max size of the queue.
    std::size_t maxSize_;
};

}

#endif // PROCESSING_QUEUE_H
