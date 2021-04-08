#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


#include "OpenCVImageDecoder.h"
#include "NvJPEGImageDecoder.h"
#include "NvJPEG2kImageDecoder.h"
#include "CUDAImage.h"
#include "Logger.h"

int main()
{
    LOGGER_INIT();
    cudaSetDevice(0);
    cv::Mat image = cv::imread("/home/valera/DIPLOM/project/3D-reconstruction/test/testImage.jpg");
    std::vector<unsigned char> jpegImage;
    std::vector<unsigned char> jpeg2kImage;
    cv::imencode(".jpg", image, jpegImage);
    cv::imencode(".jp2", image, jpeg2kImage);

    cudaStream_t nvjpegStream;
    cudaStream_t nvjpeg2kStream;
    cudaError_t status = cudaError_t::cudaSuccess;

    status = cudaStreamCreateWithFlags(&nvjpegStream, cudaStreamNonBlocking);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
    }

    status = cudaStreamCreateWithFlags(&nvjpeg2kStream, cudaStreamNonBlocking);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
    }

    Decoding::OpenCVImageDecoder openCvImageDecoder;
    openCvImageDecoder.Initialize();

    Decoding::NvJPEGImageDecoder nvJpegImageDecoder(nvjpegStream);
    nvJpegImageDecoder.Initialize();

    Decoding::NvJPEG2kImageDecoder nvJpeg2KImageDecoder(nvjpeg2kStream);
    nvJpeg2KImageDecoder.Initialize();

    cv::Mat testOpencvDecoderJpeg;
    cv::Mat testOpencvDecoderJpeg2K;
    cv::Mat testNvJpegDecoderMat;
    cv::Mat testNvJpeg2kDecoderMat;

    openCvImageDecoder.Decode(jpegImage.data(), jpegImage.size(), testOpencvDecoderJpeg);

    auto t7 = std::chrono::high_resolution_clock::now();
    openCvImageDecoder.Decode(jpeg2kImage.data(), jpeg2kImage.size(), testOpencvDecoderJpeg2K);
    auto t8 = std::chrono::high_resolution_clock::now();

    DataStructures::CUDAImage testNvJpegCUDAImage;
    DataStructures::CUDAImage testNvJpeg2kCUDAImage;

    auto t1 = std::chrono::high_resolution_clock::now();
    nvJpegImageDecoder.Decode(jpegImage.data(), jpegImage.size(), testNvJpegCUDAImage);
    cudaStreamSynchronize(nvjpegStream);
    auto t2 = std::chrono::high_resolution_clock::now();

    testNvJpegCUDAImage.MoveToCvMatAsync(testNvJpegDecoderMat, nvjpegStream);

    auto t3 = std::chrono::high_resolution_clock::now();
    nvJpeg2KImageDecoder.Decode(jpeg2kImage.data(), jpeg2kImage.size(), testNvJpeg2kCUDAImage);
    cudaStreamSynchronize(nvjpeg2kStream);
    auto t4 = std::chrono::high_resolution_clock::now();

    testNvJpeg2kCUDAImage.MoveToCvMatAsync(testNvJpeg2kDecoderMat, nvjpeg2kStream);

    cudaStreamSynchronize(nvjpegStream);
    cudaStreamSynchronize(nvjpeg2kStream);

    std::cout << "nvjpeg " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
    std::cout << "nvjpeg2k " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << std::endl;
    std::cout << "opencv jpeg2k " << std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count() << std::endl;

    auto t9 = std::chrono::high_resolution_clock::now();
    nvJpeg2KImageDecoder.Decode(jpeg2kImage.data(), jpeg2kImage.size(), testNvJpeg2kCUDAImage);
    cudaStreamSynchronize(nvjpeg2kStream);
    auto t10 = std::chrono::high_resolution_clock::now();
    std::cout << "nvjpeg2k " << std::chrono::duration_cast<std::chrono::milliseconds>(t10 - t9).count() << std::endl;

    cv::imshow("Source JPEG", image);
    cv::imshow("Test OpenCV JPEG", testOpencvDecoderJpeg);
    cv::imshow("Test OpenCV JPEG2K", testOpencvDecoderJpeg2K);
    cv::imshow("Test NvJPEG", testNvJpegDecoderMat);
    cv::imshow("Test nvjpeg2k", testNvJpeg2kDecoderMat);
    cv::waitKey(0);

    status = cudaStreamDestroy(nvjpegStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
    }

    status = cudaStreamDestroy(nvjpeg2kStream);
    if(status != cudaError_t::cudaSuccess)
    {
        LOG_ERROR() << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
    }

    LOGGER_FREE();

    return 0;
}

