#include <unordered_map>
#include <boost/stacktrace.hpp>

#include "StackTraceDumper.h"
#include "Logger.h"


const static std::unordered_map<int, const std::string> signalNames =
        {
                { SIGABRT, "SIGABRT" },
                { SIGALRM, "SIGALRM" },
                { SIGBUS, "SIGBUS" },
                { SIGCHLD, "SIGCHLD" },
                { SIGCONT, "SIGCONT" },
                { SIGFPE, "SIGFPE" },
                { SIGHUP, "SIGHUP" },
                { SIGILL, "SIGILL" },
                { SIGINT, "SIGINT" },
                { SIGKILL, "SIGKILL" },
                { SIGPIPE, "SIGPIPE" },
                { SIGQUIT, "SIGQUIT" },
                { SIGSEGV, "SIGSEGV" },
                { SIGSTOP, "SIGSTOP" },
                { SIGTERM, "SIGTERM" },
                { SIGTSTP, "SIGTSTP" },
                { SIGTTIN, "SIGTTIN" },
                { SIGTTOU, "SIGTTOU" },
                { SIGUSR1, "SIGUSR1" },
                { SIGUSR2, "SIGUSR2" },
                { SIGPOLL, "SIGPOLL" },
                { SIGPROF, "SIGPROF" },
                { SIGSYS, "SIGSYS" },
                { SIGTRAP, "SIGTRAP" },
                { SIGURG, "SIGURG" },
                { SIGVTALRM, "SIGVTALRM" },
                { SIGXCPU, "SIGXCPU" },
                { SIGXFSZ, "SIGXFSZ"},
                { SIGWINCH,"SIGWINCH"}
        };

namespace Utils
{

void StackTraceDumper::SignalHandler(int signum)
{
    LOG_FATAL() << "Received signal " << signalNames.at(signum);
    LOG_FATAL() << "Stack trace:" << std::endl << boost::stacktrace::stacktrace();
    exit(EXIT_FAILURE);
}

void StackTraceDumper::ProcessSignal(int signum)
{
    signal(signum, &StackTraceDumper::SignalHandler);
}

}