#include "StackTraceDumper.h"
#include "Logger.h"

#include "JsonConfig.h"

#include <iostream>
#include <filesystem>

int main(int argc, char** argv, char** env)
{
    LOGGER_INIT();

    Utils::StackTraceDumper::ProcessSignal(SIGABRT);
    Utils::StackTraceDumper::ProcessSignal(SIGSEGV);

    Config::JsonConfig c;

    auto test = c.Dump();

    std::cout << test << std::endl;

    LOGGER_FREE();

    return 0;
}
