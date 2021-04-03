/**
 * @file EndlessThread.h.
 *
 * @brief
 */

#ifndef ENDLESS_THREAD_H
#define ENDLESS_THREAD_H

#include <thread>
#include <condition_variable>
#include <functional>
#include <atomic>

#include "IThread.h"

/**
 * @namespace Processing
 *
 * @brief
 */
namespace Processing
{

/**
 * @class EndlessThread
 *
 * @brief
 */
class EndlessThread final : public IThread
{

public:

    /**
     * @brief
     */
    EndlessThread();

    /**
     * @brief
     */
    ~EndlessThread() override;

    /**
     * @brief
     */
    void Start() override;

    /**
     * @brief
     */
    void Stop() override;

    /**
     * @brief
     */
    void Destroy();

    /**
     * @brief
     *
     * @return
     */
    bool IsStarted() override;

    /**
     * @brief
     *
     * @param function
     */
    void SetExecutableFunction(std::function<void ()> function);

private:

    /**
     * @brief
     */
    void Execute();

    /**
     * @brief
     */
    void ExecuteInternal() override;

    ///
    std::thread thread_;

    ///
    std::mutex startMutex_;

    ///
    std::atomic_bool isStarted_;

    ///
    std::atomic_bool needToStop_;

    ///
    std::atomic_bool needToStart_;

    ///
    std::atomic_bool isDestroyed_;

    /// The condition variable
    std::condition_variable startCondition_;

    ///
    std::function<void ()> executableFunction_;

};

}

#endif // ENDLESS_THREAD_H
