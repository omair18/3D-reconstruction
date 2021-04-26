#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_elements_kernel(unsigned char* ptr, int elementsCount)
{
    for(int i = 0; i < elementsCount; i++)
        printf("%d ", (int)ptr[i]);
    printf("\n\0");
}

void print_elements_api(int blocks, int threads, int elementsCount, unsigned char* devPtr)
{
    print_elements_kernel<<<blocks, threads>>>(devPtr, elementsCount);
}

__global__ void elementwise_divide_float(const float divider, float* gpuData, size_t width, size_t pitch, size_t channels, unsigned int taskSize)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / (width * channels);
    const unsigned int elementColumn = threadId - (elementRow * width * channels);
    if(threadId < taskSize)
    {
        gpuData[elementRow * pitch + elementColumn] /= divider;
    }
}

void elementwise_divide_float_api(float divider, float* gpuData, size_t width, size_t height, size_t pitch, size_t channels, void* cudaStream)
{
    // 1 thread - 1 element
    // 1 block - 512 threads
    const unsigned int taskSize = width * height * channels;

    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    elementwise_divide_float<<<gridSize, blockSize, 0, (cudaStream_t)cudaStream>>>(divider, gpuData, width, pitch / sizeof(float), channels, taskSize);

}

__global__ void compute_gradients_kernel(float* dataLx, float* dataLy, float* resultData, size_t width, size_t pitch, size_t channels, unsigned int taskSize)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / (width * channels);
    const unsigned int elementColumn = threadId - (elementRow * width * channels);

    if(threadId < taskSize)
    {
        float dx = dataLx[elementRow * pitch + elementColumn];
        float dy = dataLy[elementRow * pitch + elementColumn];
        float grad = sqrtf(dx * dx + dy * dy);
        resultData[elementRow * pitch + elementColumn] = grad;
    }
}

void compute_gradients_api(float* dataLx, float* dataLy, float* resultData, size_t width, size_t height, size_t pitch, size_t channels, void* cudaStream)
{
    // 1 thread - 1 element
    // 1 block - 512 threads
    const unsigned int taskSize = width * height * channels;

    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    compute_gradients_kernel<<<gridSize, blockSize, 0, (cudaStream_t)cudaStream>>>(dataLx, dataLy, resultData, width, pitch / sizeof(float), channels, taskSize);
}

__global__ void preprocess_histogram_kernel(float* data, size_t width, size_t pitch, int nbins, float maxGradient, unsigned int taskSize)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / width ;
    const unsigned int elementColumn = threadId - (elementRow * width);
    const auto fnbins = (float)nbins;
    if(threadId < taskSize)
    {
        float value = data[elementRow * pitch + elementColumn];
        float processedValue = value >= 0 ? (std::floor((value / maxGradient) * fnbins)) : fnbins + 1;

        if (processedValue == fnbins)
        {
            processedValue -= 1;
        }

        if (processedValue == fnbins + 1)
        {
            processedValue = fnbins;
        }
        data[elementRow * pitch + elementColumn] = processedValue;
    }
}

void preprocess_histogram_api(float* data, size_t width, size_t height, size_t pitch, int nbins, float maxGradient, void* cudaStream)
{
    // 1 thread - 1 element
    // 1 block - 512 threads
    const unsigned int taskSize = width * height;

    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    preprocess_histogram_kernel<<<gridSize, blockSize, 0, (cudaStream_t)cudaStream>>>(data, width, pitch / sizeof(float), nbins, maxGradient, taskSize);
}