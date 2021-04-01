#include "EndlessThread.h"
#include "Logger.h"

namespace Processing
{

EndlessThread::EndlessThread() :
isStarted_(false),
needToStop_(false),
needToStart_(false),
isDestroyed_(false)
{
    thread_ = std::thread(&EndlessThread::Execute, this);
    SetExecutableFunction([&]()
    {
        LOG_WARNING() << "Thread executable function is not set.";
    });
}

EndlessThread::~EndlessThread()
{
    Destroy();
}

void EndlessThread::ExecuteInternal()
{
    LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " was created.";
    while (true)
    {
        if (isDestroyed_)
        {
            return;
        }
        LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " is waiting for start command ...";
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

        while (!needToStop_)
        {
            executableFunction_();
        }
        isStarted_ = false;
        needToStop_ = false;
        LOG_TRACE() << "Thread with id " << std::this_thread::get_id() << " was stopped.";
        if (isDestroyed_)
        {
            return;
        }
    }
}

void EndlessThread::Start()
{
    needToStart_ = true;
    startCondition_.notify_one();
}

void EndlessThread::Stop()
{
    needToStop_ = true;
}

void EndlessThread::Destroy()
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

bool EndlessThread::IsStarted()
{
    return isStarted_;
}

void EndlessThread::SetExecutableFunction(std::function<void()> function)
{
    executableFunction_ = std::move(function);
}

void EndlessThread::Execute()
{
    return ExecuteInternal();
}

}