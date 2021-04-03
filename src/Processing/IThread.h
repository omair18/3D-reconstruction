/**
 * @file IThread.h.
 *
 * @brief
 */

#ifndef INTERFACE_THREAD_H
#define INTERFACE_THREAD_H

/**
 * @namespace Processing
 *
 * @brief
 */
namespace Processing
{

/**
 * @class IThread
 *
 * @brief
 */
class IThread
{

public:

    /**
     * @brief
     */
    IThread() = default;

    /**
     * @brief
     */
    virtual ~IThread() = default;

    /**
     * @brief
     */
    virtual void Start() = 0;

    /**
     * @brief
     */
    virtual void Stop() = 0;

    /**
     * @brief
     *
     * @return
     */
    virtual bool IsStarted() = 0;

protected:

    /**
     * @brief
     */
    virtual void ExecuteInternal() = 0;

};

}

#endif // INTERFACE_THREAD_H
