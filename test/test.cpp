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

    LOG_TRACE() << "Creating a.";
    DataStructures::CUDAImageDescriptor a;
    a.GetCUDAImage()->Allocate(10, 10, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, false);

    LOG_TRACE() << "Creating c.";
    DataStructures::CUDAImageDescriptor c;
    c.GetCUDAImage()->Allocate(10, 10, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, false);

    LOG_TRACE() << "Creating b.";
    std::vector<DataStructures::CUDAImageDescriptor> b;
    LOG_TRACE() << "b cap = " << b.capacity();

    LOG_TRACE() << "b push std::move(a).";
    b.push_back(std::move(a));
    LOG_TRACE() << "b cap = " << b.capacity();

    LOG_TRACE() << "b push c.";
    b.push_back(std::move(c));
    LOG_TRACE() << "b cap = " << b.capacity();

    LOG_TRACE() << "b reserve 10.";
    b.reserve(10);
    LOG_TRACE() << "b cap = " << b.capacity();

    //test();

    //LOGGER_FREE();
    return 0;
}
