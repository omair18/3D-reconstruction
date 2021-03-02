#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <nppi_geometry_transforms.h>
#include <filesystem>
#include <iostream>
#include <chrono>

#include "cuda_test_kernels.h"

int main()
{
    std::filesystem::path imagesPath = "/home/valera/3/datasets/odm_data_aukerman/images";

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

        cv::cuda::GpuMat cvGpuMat;
        cvGpuMat.upload(image);

        uchar* cudaImage = nullptr;
        size_t cudaImageWidth = image.cols;
        size_t cudaImageHeight = image.rows;
        size_t cudaImageChannels = image.channels();
        size_t cudaElementSize = sizeof(uchar);
        size_t cudaImagePitch = 0;

        cudaError_t status = cudaError_t::cudaSuccess;
        status = cudaMallocPitch(&cudaImage,
                                 &cudaImagePitch,
                                 cudaImageWidth * cudaImageChannels * cudaElementSize,
                                 cudaImageHeight);
        status = cudaMemcpy2D(cudaImage,
                              cudaImagePitch,
                              image.data,
                              image.channels() * image.cols * cudaElementSize,
                              cudaImageWidth * cudaImageChannels * cudaElementSize,
                              cudaImageHeight,
                              cudaMemcpyHostToDevice);


        cv::Size resizedSize(800, 600);
        cv::Mat resizedImage(resizedSize, CV_8UC3);

        auto cvResizeBegin = std::chrono::high_resolution_clock::now();
        cv::resize(image, resizedImage, resizedSize);
        auto cvResizeEnd = std::chrono::high_resolution_clock::now();

        cv::cuda::GpuMat resizedGpu(resizedSize, CV_8UC3);

        auto cvGpuResizeBegin = std::chrono::high_resolution_clock::now();
        cv::cuda::resize(cvGpuMat, resizedGpu, resizedSize);
        auto cvGpuResizeEnd = std::chrono::high_resolution_clock::now();

        NppiSize srcSize = { .width = (int)cudaImageWidth, .height = (int)cudaImageHeight };
        NppiRect srcRoi = { .x = 0, .y = 0, .width = srcSize.width, .height = srcSize.height };
        NppiSize dstSize = { .width = resizedSize.width, .height = resizedSize.height };
        NppiRect dstRoi = { .x = 0, .y = 0, .width = dstSize.width, .height = dstSize.height };
        NppStatus st = NppStatus::NPP_NO_ERROR;

        uchar* cudaImageResized = nullptr;
        size_t cudaImageResizedWidth = resizedSize.width;
        size_t cudaImageResizedHeight = resizedSize.height;
        size_t cudaImageResizedChannels = image.channels();
        size_t cudaImageResizedSize = sizeof(uchar);
        size_t cudaImageResizedPitch = 0;

        status = cudaMallocPitch(&cudaImageResized,
                                 &cudaImageResizedPitch,
                                 cudaImageResizedWidth * cudaImageResizedChannels * cudaImageResizedSize,
                                 cudaImageResizedHeight);

        cv::cuda::GpuMat test(resizedSize, CV_8UC3);

        auto nppCvResizeBegin = std::chrono::high_resolution_clock::now();
        st = nppiResize_8u_C3R(cvGpuMat.data, cvGpuMat.step, srcSize, srcRoi, test.data, test.step, dstSize, dstRoi, NPPI_INTER_LINEAR);
        auto nppCvResizeEnd = std::chrono::high_resolution_clock::now();

        auto nppResizeBegin = std::chrono::high_resolution_clock::now();
        st = nppiResize_8u_C3R(cudaImage, cudaImagePitch, srcSize, srcRoi,
                               cudaImageResized, cudaImageResizedPitch, dstSize, dstRoi, NPPI_INTER_LINEAR);
        auto nppResizeEnd = std::chrono::high_resolution_clock::now();

        cv::Mat cvGpuResizedHost;
        resizedGpu.download(cvGpuResizedHost);

        uchar cudaImageResizedHost[cudaImageResizedWidth * cudaImageResizedHeight * cudaImageChannels];

        status = cudaMemcpy2D(cudaImageResizedHost, cudaImageResizedWidth * cudaImageChannels * sizeof(uchar),
                     cudaImageResized, cudaImageResizedPitch * sizeof(uchar), resizedSize.width * cudaImageChannels* sizeof(uchar), resizedSize.height,
                     cudaMemcpyKind::cudaMemcpyDeviceToHost);

        status = cudaFree(cudaImage);
        status = cudaFree(cudaImageResized);

        cv::Mat cudaResizedHostMat(resizedSize, CV_8UC3, cudaImageResizedHost);

        cv::Mat cvNppiResizedHostMat;
        test.download(cvNppiResizedHostMat);

        cv::imshow("opencv", resizedImage);
        cv::imshow("opencv_gpu", cvGpuResizedHost);
        cv::imshow("nppi", cudaResizedHostMat);
        cv::imshow("opencv_nppi", cvNppiResizedHostMat);

        std::cout << "Opencv time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(cvResizeEnd - cvResizeBegin)).count() << " ns" << std::endl;
        std::cout << "Opencv CUDA time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(cvGpuResizeEnd - cvGpuResizeBegin)).count() << " ns" << std::endl;
        std::cout << "CUDA nppi time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(nppResizeEnd - nppResizeBegin)).count() << " ns"  << std::endl;
        std::cout << "CUDA nppi opencv time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(nppCvResizeEnd - nppCvResizeBegin)).count() << " ns"  << std::endl;

        cv::waitKey(10);
    }

    return 0;
}