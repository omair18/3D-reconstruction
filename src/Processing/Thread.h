#ifndef THREAD_H
#define THREAD_H

#include <thread>
#include <condition_variable>
#include <functional>
#include <atomic>


namespace Processing
{

class Thread
{
public:

    Thread();

    virtual ~Thread();

    void Start();

    void Stop();

    void Destroy();

    bool IsStarted();

    void SetExecutableFunction(std::function<void ()> function);

protected:

    virtual void ExecuteInternal();

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

private:

    void Execute();

};

}

#endif // THREAD_H
