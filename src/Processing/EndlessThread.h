#ifndef ENDLESS_THREAD_H
#define ENDLESS_THREAD_H

#include <thread>
#include <condition_variable>
#include <functional>
#include <atomic>

#include "IThread.h"

namespace Processing
{

class EndlessThread final : public IThread
{

public:

    EndlessThread();

    ~EndlessThread() override;

    void Start() override;

    void Stop() override;

    void Destroy();

    bool IsStarted() override;

    void SetExecutableFunction(std::function<void ()> function);

private:

    void Execute();

    void ExecuteInternal() override;

    /// Thread
    std::thread thread_;

    std::mutex startMutex_;

    std::atomic_bool isStarted_;

    std::atomic_bool needToStop_;

    std::atomic_bool needToStart_;

    std::atomic_bool isDestroyed_;

    /// The condition variable
    std::condition_variable startCondition_;

    std::function<void ()> executableFunction_;

};

}

#endif // ENDLESS_THREAD_H
