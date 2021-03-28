#include "EndlessThread.h"
#include "Logger.h"

namespace Processing
{

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
}