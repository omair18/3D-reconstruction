#ifndef INTERFACE_THREAD_H
#define INTERFACE_THREAD_H

namespace Processing
{

class IThread
{

public:

    IThread() = default;

    virtual ~IThread() = default;

    virtual void Start() = 0;

    virtual void Stop() = 0;

    virtual bool IsStarted() = 0;

protected:

    virtual void ExecuteInternal() = 0;

};

}

#endif // INTERFACE_THREAD_H
