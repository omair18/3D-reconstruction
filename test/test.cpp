#include <iostream>
#include <tuple>
#include <boost/static_string.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include "ImageDecoderFactory.h"
#include "CUDAImage.h"
#include "ProcessingQueue.h"
#include "ProcessingData.h"
#include "Logger.h"

int main()
{
    LOGGER_INIT();
    std::string a = "a,b,c,d,";
    a = a.substr(0, a.size() - 1);

    std::vector<int> d = {1, 2, 3, 4};
    std::apply([](auto&&... args)
    {
        ((std::cout << args), ...);
    },
    std::make_tuple("This is ", "Sparta!!!"));

    LOGGER_FREE();
    return 0;
}
