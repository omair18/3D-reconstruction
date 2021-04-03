#include <cuda_runtime.h>
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
    openCvImageDecoder.Decode(jpeg2kImage.data(), jpeg2kImage.size(), testOpencvDecoderJpeg2K);

    DataStructures::CUDAImage testNvJpegCUDAImage;
    DataStructures::CUDAImage testNvJpeg2kCUDAImage;

    nvJpegImageDecoder.Decode(jpegImage.data(), jpegImage.size(), testNvJpegCUDAImage);

    testNvJpegCUDAImage.MoveToCvMatAsync(testNvJpegDecoderMat, nvjpegStream);

    nvJpeg2KImageDecoder.Decode(jpeg2kImage.data(), jpeg2kImage.size(), testNvJpeg2kCUDAImage);

    testNvJpeg2kCUDAImage.MoveToCvMatAsync(testNvJpeg2kDecoderMat, nvjpeg2kStream);

    cudaStreamSynchronize(nvjpegStream);
    cudaStreamSynchronize(nvjpeg2kStream);

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

