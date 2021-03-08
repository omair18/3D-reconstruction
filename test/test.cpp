#include <iostream>
#include <tuple>
#include <boost/static_string.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>

int main()
{
    std::apply([](auto&&... args)
    {
        ((std::cout << args), ...);
    },
    std::make_tuple("This is ", "Sparta!!!"));

    cv::Mat image = cv::imread("/home/valera/Photo/5/frames/0.jpg");

    cv::cuda::GpuMat testMat(image);

    unsigned char * tptr = testMat.data;
    testMat.data = nullptr;
    size_t pitch = testMat.step;
    testMat.step = 0;
    size_t w, h;
    w = testMat.cols;
    testMat.cols = 0;

    h = testMat.rows;
    testMat.rows = 0;

    return 0;
}
