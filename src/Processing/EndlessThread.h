#ifndef ENDLESS_THREAD_H
#define ENDLESS_THREAD_H

#include "Thread.h"

namespace Processing
{

class EndlessThread : public Thread
{

public:

    EndlessThread() = default;

    ~EndlessThread() override;
private:
    void ExecuteInternal() override;


};

}

#endif // ENDLESS_THREAD_H
