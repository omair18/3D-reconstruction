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

int main()
{
    std::apply([](auto&&... args)
    {
        ((std::cout << args), ...);
    },
    std::make_tuple("This is ", "Sparta!!!"));



    return 0;
}
