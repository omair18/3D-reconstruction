#include <nppi_filtering_functions.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main()
{
    cv::Mat image = cv::imread("/home/valera/Photo/5/frames2/0.jpg");
    cv::imshow("test", image);
    cv::waitKey(0);

    size_t width = image.cols;
    size_t height = image.rows;
    size_t pitch = 0;
    size_t channels = image.channels();
    size_t elementSize = image.elemSize1();
    void* gpuData = nullptr;

    auto status = cudaMallocPitch(&gpuData, &pitch, width * channels * elementSize, height);

    status = cudaMemcpy2D(gpuData, pitch, image.data, width * channels * elementSize, width * channels * elementSize, height, cudaMemcpyKind::cudaMemcpyHostToDevice);


    status = cudaFree(gpuData);
    return 0;
}
