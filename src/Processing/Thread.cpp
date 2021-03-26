#include <chrono>

#include "Thread.h"
#include "Logger.h"

namespace Processing
{

bool Thread::Start()
{
    std::lock_guard<std::mutex> lock1(mutexInternal_);
    std::lock_guard<std::mutex> lock2(stateMutex_);
    if (!isStarted_)
    {
        isStarted_ = true;
        if (StartInternal())
        {
            return true;
        }
        else
        {
            LOG_WARNING() << "Failed to start thread with id " << thread_.get_id() << ".";
            isStarted_ = false;
            return false;
        }
    }
    else
    {
        LOG_WARNING() << "Thread with id " << thread_.get_id() << " is already started.";
        return true;
    }
}

bool Thread::Stop()
{
    std::lock_guard<std::mutex> lock1(mutexInternal_);
    {
        std::lock_guard<std::mutex> lock2(stateMutex_);
        if (!isStarted_)
        {
            return true;
        }

        if (!StopInternal())
        {
            return false;
        }

        isStarted_ = false;
    }

    WaitForStop();
    return true;
}

bool Thread::Restart()
{
    Stop();
    return Start();
}

bool Thread::IsStarted()
{
    std::lock_guard<std::mutex> lock2(stateMutex_);
    return isStarted_;
}

bool Thread::StartInternal()
{
    thread_ = std::thread(&Thread::Routine, this);
    return true;
}

bool Thread::StopInternal()
{
    return true;
}

bool Thread::WaitForStop()
{
    stopCondition_.notify_one();
    thread_.join();
    return true;
}

void Thread::Routine()
{
    LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " was started.";

    while (isStarted_)
    {
        std::unique_lock<std::mutex> lock(stateMutex_);
        if (stopCondition_.wait_for(lock, std::chrono::milliseconds(100)) == std::cv_status::no_timeout)
        {
            break;
        }
    }

    LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " was stopped.";
}

}