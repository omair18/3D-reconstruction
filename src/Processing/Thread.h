#ifndef THREAD_H
#define THREAD_H

#include <thread>
#include <condition_variable>


namespace Processing
{

class Thread
{
public:

    bool Start();

    bool Stop();

    bool Restart();

    bool IsStarted();

private:

    virtual bool StartInternal();

    virtual bool StopInternal();

    virtual bool WaitForStop();

    virtual void Routine();

    /// Thread
    std::thread thread_;

    /// The condition variable
    std::condition_variable stopCondition_;

    /** The internal mutex */
    std::mutex mutexInternal_; // guards start/stop process with waiting

    bool isStarted_ = false;

    /** The mutex */
    std::mutex stateMutex_; // guards isStarted_ variable

};

}

#endif // THREAD_H
