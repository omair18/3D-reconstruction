#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <utility>

namespace DataStructures
{

/**
 * @class ThreadSafeQueue
 *
 * @brief
 *
 * @tparam Type
 */
template<class Type>
class ThreadSafeQueue
{
public:

    /**
     * @brief
     *
     * @param
     * @param
     */
    ThreadSafeQueue(std::string name, int maxSize) :
    name_(std::move(name)), maxSize_(maxSize)
    {

    };

    /**
     * @brief
     */
    ~ThreadSafeQueue() = default;

    /**
     * @brief Puts an element to the queue.
     *
     * @param element - Element to put.
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
     * @brief
     *
     * @param element
     */
    void PutAsync(const Type& element)
    {
        std::thread([this, element]()
                    {
                        {
                            std::unique_lock<std::mutex> lock(this->dequeMutex_);
                            this->dequeNotFullCV_.wait(lock, [&](){
                                return this->deque_.size() < this->maxSize_;
                            });
                            this->deque_.push_back(element);
                        }
                        this->dequeNotEmptyCV_.notify_one();
                    }).detach();
    }

    /**
     * @brief Put a batch of elements in the queue
     *
     * @param batch of elements
     */
    template<template <class...> class Container>
    void PutBatch(const Container<Type>& elements)
    {
        {
            std::unique_lock<std::mutex> lock(dequeMutex_);
            dequeNotFullCV_.wait(lock, [&](){
                return deque_.size() + elements.size() <= maxSize_;
            });
            deque_.insert(deque_.end(), elements.begin(), elements.end());
        }
        dequeNotEmptyCV_.notify_one();
    }

    /**
     * @brief
     *
     * @tparam Container
     * @param elements
     */
    template<template <class...> class Container>
    void PutBatchAsync(const Container<Type>& elements)
    {
        std::thread([this, elements]()
                    {
                        {
                            std::unique_lock<std::mutex> lock(this->dequeMutex_);
                            this->dequeNotFullCV_.wait(lock, [&](){
                                return this->deque_.size() + elements.size() <= this->maxSize_;
                            });
                            this->deque_.insert(deque_.end(), elements.begin(), elements.end());
                        }
                        this->dequeNotEmptyCV_.notify_one();
                    }).detach();
    }

    /**
     * @brief Get an element from the queue
     *
     * @return element
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
     * @brief Try to get a product from the queue
     *
     * @param element
     * @return True if it succeeds, false if it fails.
     */
    bool TryGet(Type& element)
    {
        std::lock_guard<std::mutex> lock(dequeMutex_);
        if (!deque_.empty())
        {
            element = deque_.front();
            deque_.pop_front();
            dequeNotFullCV_.notify_one();
            return true;
        }
        else
        {
            return false;
        }
    }

    /**
     * @brief Get a batch of elements from the queue with at least one element with timeout
     *
     * @param elements  Container for receiving elements
     * @param batchSize Maximum container size
     * @param timeout   Timeout in seconds
     */
    template<template <class...> class Container>
    void GetBatch(Container<Type>& elements, size_t batchSize, float timeout)
    {
        std::unique_lock<std::mutex> lock(dequeMutex_);
        dequeNotEmptyCV_.wait(lock, [&](){
            return !deque_.empty();
        });
        elements.push_back(deque_.front());
        deque_.pop_front();
        dequeNotFullCV_.notify_one();

        const auto timeoutPoint = std::chrono::system_clock::now() + std::chrono::milliseconds(static_cast<int>(timeout * 1000));

        while(elements.size() != batchSize)
        {
            dequeNotEmptyCV_.wait_until(lock, timeoutPoint, [&](){
                return !deque_.empty();
            });

            if (deque_.empty())
            {
                break;
            }
            else
            {
                auto needCount = batchSize - elements.size();

                if (deque_.size() < needCount)
                {
                    elements.insert(elements.end(), deque_.begin(), deque_.begin() + deque_.size());
                    deque_.clear();
                }
                else
                {
                    elements.insert(elements.end(), deque_.begin(), deque_.begin() + needCount);
                    deque_.erase(deque_.begin(), deque_.begin() + needCount);
                }

                dequeNotFullCV_.notify_one();
            }
        }
    }

    /**
     * @brief Clear the queue
     */
    inline void Clear()
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
    /// The queue
    std::deque<Type> deque_;

    ///The mutex of the queue
    std::mutex dequeMutex_;

    /// The conditional variable of the queue
    std::condition_variable dequeNotEmptyCV_;

    /// The conditional variable of the queue
    std::condition_variable dequeNotFullCV_;

    /// The name of the queue
    std::string name_;

    /// The max size of the queue
    std::size_t maxSize_;
};

}

#endif // THREAD_SAFE_QUEUE_H
