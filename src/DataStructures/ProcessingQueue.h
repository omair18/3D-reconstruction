#ifndef PROCESSING_QUEUE_H
#define PROCESSING_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <utility>

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
     * @param
     * @param
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
