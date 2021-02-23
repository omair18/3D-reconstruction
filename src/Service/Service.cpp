#include <iostream>

#include "ServiceSDK.h"

int main(int argc, char** argv)
{
    int exitCode = EXIT_SUCCESS;
    Service::ServiceSDK serviceSdk(argc, argv);
    try
    {
        serviceSdk.Initialize();
    }
    catch (std::exception& exception)
    {
        std::clog << exception.what() << std::endl;
        exitCode = EXIT_FAILURE;
    }
    return exitCode;
}
