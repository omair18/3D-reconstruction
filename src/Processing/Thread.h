/**
 * @file Thread.h.
 *
 * @brief
 */

#ifndef THREAD_H
#define THREAD_H

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
 * @class Thread
 *
 * @brief
 */
class Thread final : public IThread
{

public:

    /**
     * @brief
     */
    Thread();

    /**
     * @brief
     */
    ~Thread() override;

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

    ///
    std::condition_variable startCondition_;

    ///
    std::function<void ()> executableFunction_;

};

}

#endif // THREAD_H
