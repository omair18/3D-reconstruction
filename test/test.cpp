#include <iostream>
#include <filesystem>
#include <tuple>
#include <boost/static_string.hpp>
#include <boost/bind/bind.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>
#include "ImageDecoderFactory.h"
#include "CUDAImage.h"
#include "ProcessingQueue.h"
#include "ProcessingData.h"
#include "Logger.h"
#include "Thread.h"
#include "EndlessThread.h"

void test()
{
    Processing::EndlessThread thread;

    thread.SetExecutableFunction([]()
                                 {
                                     for (int i = 0; i < 10; ++i)
                                     {
                                         std::cout << i << std::endl;
                                         std::this_thread::sleep_for(std::chrono::milliseconds(100));
                                     }
                                 });
};

int main()
{
    LOGGER_INIT();


    test();

    LOGGER_FREE();
    return 0;
}
