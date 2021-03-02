#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nppi_filtering_functions.h>
#include <filesystem>
#include <iostream>
#include <chrono>

#include "cuda_test_kernels.h"

int main()
{
    float kernel_[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    cv::Mat kernel(cv::Size(3, 3), CV_32F, kernel_);
    std::filesystem::path imagesPath = "/home/valera/Photo/5/frames2";

    for (auto& file : std::filesystem::directory_iterator(imagesPath))
    {
        cv::Mat image = cv::imread(file.path().string());
        if(!image.empty())
        {
            //std::cout << "Successfully read " << file.path().string() << std::endl;
        }
        else
        {
            //std::cout << "Failed to read " << file.path().string() << std::endl;
            continue;
        }
        //cv::resize(image, image, cv::Size(800, 600));
        cv::Size imageSize(image.cols, image.rows);


        float* cudaFilter;
        cudaMalloc(&cudaFilter, 9 * sizeof(float));
        cudaMemcpy(cudaFilter, kernel_, 9 * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

        cv::Mat filtered;

        auto opencvBegin = std::chrono::high_resolution_clock::now();
        cv::filter2D(image, filtered, -1, kernel);
        auto opencvEnd = std::chrono::high_resolution_clock::now();

        NppiSize nppImageSize {.width = image.cols, .height = image.rows};
        NppiSize nppKernelSize {.width = 3, .height = 3};
        NppiPoint nppAnchor { .x = 0, .y = 0 };
        NppiPoint displacement { .x = 0, .y = 0 };

        unsigned char* cudaImage = nullptr;
        size_t cudaImageWidth = imageSize.width;
        size_t cudaImageHeight = imageSize.height;
        size_t cudaImageChannels = image.channels();
        size_t cudaImagePitch = 0;

        cudaError_t cudaStatus = cudaError_t::cudaSuccess;
        cudaStatus = cudaMallocPitch(&cudaImage, &cudaImagePitch, cudaImageWidth * cudaImageChannels * sizeof(unsigned char),
                        cudaImageHeight);
        cudaStatus = cudaMemcpy2D(cudaImage, cudaImagePitch, image.data, image.cols * image.channels() * sizeof(unsigned char),
                     image.cols * image.channels() * sizeof(unsigned char), image.rows, cudaMemcpyKind::cudaMemcpyHostToDevice);

        unsigned char* cudaResultImage = nullptr;
        size_t cudaResultImageWidth = imageSize.width;
        size_t cudaResultImageHeight = imageSize.height;
        size_t cudaResultImageChannels = image.channels();
        size_t cudaResultImagePitch = 0;

        cudaStatus = cudaMallocPitch(&cudaResultImage, &cudaResultImagePitch, cudaResultImageWidth * cudaResultImageChannels * sizeof(unsigned char),
                                     cudaResultImageHeight);


        NppStatus status = NppStatus::NPP_NO_ERROR;
        auto nppBegin = std::chrono::high_resolution_clock::now();
        status = nppiFilterBorder32f_8u_C3R(cudaImage, cudaImagePitch, nppImageSize, displacement, cudaResultImage,
                                   cudaImagePitch, nppImageSize, cudaFilter, nppKernelSize, nppAnchor, NppiBorderType::NPP_BORDER_REPLICATE);
        auto nppEnd = std::chrono::high_resolution_clock::now();

        auto err = cudaGetLastError();
        std::cout << err << " " << cudaGetErrorString(err) << std::endl;

        uchar hostImage[cudaImageWidth * cudaImageHeight * cudaImageChannels];

        cudaStatus = cudaMemcpy2D(hostImage,
                                  cudaImageWidth * cudaImageChannels * sizeof(unsigned char),
                                  cudaResultImage,
                                  cudaResultImagePitch,
                                  cudaImageWidth * cudaImageChannels * sizeof(unsigned char),
                                  cudaImageHeight,
                                  cudaMemcpyKind::cudaMemcpyDeviceToHost);

        std::cout << err << " " << cudaGetErrorString(cudaStatus) << std::endl;

        cudaStatus = cudaFree(cudaResultImage);
        cudaStatus = cudaFree(cudaImage);
        cv::Mat cvGpuFilteredHost(imageSize, CV_8UC3, hostImage);

        cudaFree(cudaFilter);

        //cv::imshow("source", image);
        //cv::imshow("opencv", filtered);
        //cv::imshow("opencv cuda", cvGpuFilteredHost);

        std::cout << "opencv time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(opencvEnd - opencvBegin)).count() << " ns"  << std::endl;
        std::cout << "nppi time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(nppEnd - nppBegin)).count() << " ns"  << std::endl;

        //cv::waitKey(0);

    }

    return 0;
}
