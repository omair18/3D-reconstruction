#include <opencv2/core/mat.hpp>
#include <nppi_arithmetic_and_logical_operations.h>
#include <iostream>
#include <chrono>

#include "Logger.h"
#include "CUDAImage.h"
#include "cuda_test_kernels.h"

int main()
{
    LOGGER_INIT();

    cv::Mat testMat(7200, 12800, CV_32FC1, 128);
    cv::Mat testResult, testResult1, testResult2;

    DataStructures::CUDAImage image1, image2, result;

    image1.CopyFromCvMat(testMat);
    image2.CopyFromCvMat(testMat);
    result.Allocate(image1.width_, image1.height_, image1.channels_, image1.elementType_, image1.pitchedAllocation_);

    auto t1 = std::chrono::high_resolution_clock::now();
    testResult = testMat / 16;
    auto t2 = std::chrono::high_resolution_clock::now();

    NppiSize roi {.width = (int)image1.width_, .height = (int)image1.height_ };
    NppStatus status = NppStatus::NPP_NO_ERROR;

    auto t3 = std::chrono::high_resolution_clock::now();
    status = nppiDivC_32f_C1R((float*)image1.gpuData_, image1.pitch_, 16, (float*)result.gpuData_, result.pitch_, roi);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();

    auto t5 = std::chrono::high_resolution_clock::now();
    status = nppiDivC_32f_C1R((float*)image2.gpuData_, image2.pitch_, 16, (float*)image2.gpuData_, image2.pitch_, roi);
    cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();

    auto t7 = std::chrono::high_resolution_clock::now();
    elementwise_divide_float_api(16.f, (float*)image1.gpuData_, image1.width_, image1.height_, image1.pitch_, image1.channels_,
                                 nullptr);
    auto state = cudaDeviceSynchronize();
    auto t8 = std::chrono::high_resolution_clock::now();

    result.MoveToCvMat(testResult1);

    image2.MoveToCvMat(testResult2);

    std::cout << "T1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "ns" << std::endl;
    std::cout << "T2: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << "ns" << std::endl;
    std::cout << "T3: " << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() << "ns" << std::endl;
    std::cout << "T4: " << std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count() << "ns" << std::endl;

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            std::cout << "[" << i << ", " << j << "] OpenCV: " << (float)testResult.at<float>(i, j) << "\tNew: " << (float)testResult1.at<float>(i, j) << "\tExisting: " << (float)testResult2.at<float>(i, j) << std::endl;
        }
    }
    return 0;
}
