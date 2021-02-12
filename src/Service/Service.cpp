#include "StackTraceDumper.h"
#include "Logger.h"

int main(int argc, char** argv, char** env)
{
    LOGGER_INIT();

    Utils::StackTraceDumper::ProcessSignal(SIGABRT);
    Utils::StackTraceDumper::ProcessSignal(SIGSEGV);

    LOGGER_FREE();

    return 0;
}
