/**
 * @file StackTraceDumper.h.
 *
 * @brief Declares the stacktrace dumper class.
 */

#ifndef STACK_TRACE_DUMPER_H
#define STACK_TRACE_DUMPER_H

#include <csignal>

/**
 * @namespace Utils
 *
 * @brief Namespace of libutils library.
 */
namespace Utils
{

/**
 * @class StackTraceDumper
 *
 * @brief The class of stacktrace dumper.
 */
class StackTraceDumper
{
public:

    /**
     * @brief This method adds signum-parameter to the list of signals to process.
     *
     * @param signum - signal defined in &lt;csignal&gt; or in &lt;signal.h&gt;
     */
    static void ProcessSignal(int signum);

private:
    /**
     * @brief Default constructor.
     */
    StackTraceDumper() = default;

    /**
     * @brief Default destructor.
     */
    ~StackTraceDumper() = default;

    /**
     * @brief This method is being called when it is time to process signal, it prints stacktrace to log file with FATAL
     * severity and calls exit function with EXIT_FAILURE code.
     *
     * @param signum - signal defined in &lt;csignal&gt; or in &lt;signal.h&gt;
     */
    static void SignalHandler(int signum);
};

}
#endif // STACK_TRACE_DUMPER_H
