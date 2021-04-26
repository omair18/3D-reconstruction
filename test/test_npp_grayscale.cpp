#include <nppi_color_conversion.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat image = cv::imread("/home/valera/DIPLOM/project/3D-reconstruction/test/1.png");
    cv::resize(image, image, cv::Size(800, 800));
    cv::imshow("source", image);
    cv::waitKey(0);

    cv::Mat grayOpenCv;

    cv::cvtColor(image, grayOpenCv, cv::COLOR_BGR2GRAY);
    cv::imshow("opencv", grayOpenCv);
    cv::waitKey(0);

    size_t sourceCudaImageWidth = image.cols;
    size_t sourceCudaImageHeight = image.rows;
    size_t sourceCudaImagePitch = 0;
    size_t sourceCudaImageChannels = image.channels();
    size_t sourceCudaImageElementSize = image.elemSize1();
    Npp8u*  sourceCudaImageGpuData = nullptr;

    size_t nppiImageWidth = image.cols;
    size_t nppiImageHeight = image.rows;
    size_t nppiImagePitch = 0;
    size_t nppiImageChannels = 1;
    size_t nppiImageElementSize = sizeof(unsigned char);
    Npp8u*  nppiImageGpuData = nullptr;

    size_t nppiCustomImageWidth = image.cols;
    size_t nppiCustomImageHeight = image.rows;
    size_t nppiCustomImagePitch = 0;
    size_t nppiCustomImageChannels = 1;
    size_t nppiCustomImageElementSize = sizeof(unsigned char);
    Npp8u*  nppiCustomImageGpuData = nullptr;

    std::vector<float> customCoefficients = {0.114f, 0.587f, 0.299f};

    cudaError_t status = cudaError_t::cudaSuccess;

    status = cudaMallocPitch(&sourceCudaImageGpuData, &sourceCudaImagePitch, sourceCudaImageWidth * sourceCudaImageChannels * sourceCudaImageElementSize, sourceCudaImageHeight);
    status = cudaMallocPitch(&nppiImageGpuData, &nppiImagePitch, nppiImageWidth * nppiImageChannels * nppiImageElementSize, nppiImageHeight);
    status = cudaMallocPitch(&nppiCustomImageGpuData, &nppiCustomImagePitch, nppiCustomImageWidth * nppiCustomImageChannels * nppiCustomImageElementSize, nppiCustomImageHeight);
    status = cudaMemcpy2D(sourceCudaImageGpuData, sourceCudaImagePitch, image.data, image.cols * image.channels() * image.elemSize1(), image.cols * image.channels() * image.elemSize1(), image.rows, cudaMemcpyKind::cudaMemcpyHostToDevice);

    NppStatus nppStatus = NppStatus::NPP_NO_ERROR;
    NppiSize roi = { .width = (int)nppiImageWidth, .height = (int)nppiImageHeight };

    nppStatus = nppiRGBToGray_8u_C3C1R(sourceCudaImageGpuData, sourceCudaImagePitch, nppiImageGpuData, nppiImagePitch, roi);

    status = cudaDeviceSynchronize();

    nppStatus = nppiColorToGray_8u_C3C1R(sourceCudaImageGpuData, sourceCudaImagePitch, nppiCustomImageGpuData, nppiCustomImagePitch, roi, customCoefficients.data());

    cv::Mat nppGrayImage(nppiImageHeight, nppiImageWidth, CV_8UC1);
    cv::Mat nppGrayImageCustom(nppiImageHeight, nppiImageWidth, CV_8UC1);

    status = cudaDeviceSynchronize();

    status = cudaMemcpy2D(nppGrayImage.data, nppiImageWidth * nppiImageElementSize * nppiImageChannels, nppiImageGpuData, nppiImagePitch, nppiImageWidth * nppiImageElementSize * nppiImageChannels, nppiImageHeight, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    status = cudaMemcpy2D(nppGrayImageCustom.data, nppiCustomImageWidth * nppiCustomImageElementSize * nppiCustomImageChannels, nppiCustomImageGpuData, nppiCustomImagePitch, nppiCustomImageWidth * nppiCustomImageElementSize * nppiCustomImageChannels, nppiCustomImageHeight, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cv::imshow("nppi standard", nppGrayImage);

    cv::imshow("nppi custom", nppGrayImageCustom);

    cv::waitKey(0);

    status = cudaFree(sourceCudaImageGpuData);
    status = cudaFree(nppiImageGpuData);
    status = cudaFree(nppiCustomImageGpuData);

    return 0;
}
