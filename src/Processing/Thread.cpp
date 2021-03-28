#include <chrono>

#include "Thread.h"
#include "Logger.h"

namespace Processing
{

Thread::Thread() :
isStarted_(false),
needToStop_(false),
needToStart_(false),
isDestroyed_(false)
{
    thread_ = std::thread(&Thread::Execute, this);
    SetExecutableFunction([&]()
    {
        LOG_WARNING() << "Thread executable function is not set.";
    });
}

void Thread::Start()
{
    needToStart_ = true;
    startCondition_.notify_one();
}

void Thread::Execute()
{
    return ExecuteInternal();
}

void Thread::Stop()
{
    needToStop_ = true;
}

bool Thread::IsStarted()
{
    return isStarted_;
}

void Thread::SetExecutableFunction(std::function<void()> function)
{
    executableFunction_ = std::move(function);
}

void Thread::ExecuteInternal()
{
    LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " was created.";
    LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " is waiting for start command ...";
    if (isDestroyed_)
    {
        return;
    }
    std::unique_lock<std::mutex> lock(startMutex_);
    while (!needToStart_)
    {
        startCondition_.wait(lock);
    }
    if (isDestroyed_)
    {
        return;
    }
    isStarted_ = true;
    needToStart_ = false;
    LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " was started.";
    if(!needToStop_)
    {
        executableFunction_();
    }
    isStarted_ = false;
    needToStop_ = false;
    LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " finished execution.";

}

Thread::~Thread()
{
    Destroy();
}

void Thread::Destroy()
{
    if (!isDestroyed_)
    {
        isDestroyed_ = true;
        Stop();
        startCondition_.notify_one();
        if (thread_.joinable())
        {
            thread_.join();
        }
        LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " was destroyed.";
    }
}

}