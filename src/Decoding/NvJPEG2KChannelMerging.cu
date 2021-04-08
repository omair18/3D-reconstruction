#include <cuda_runtime.h>

#include "NvJPEG2KChannelMerging.h"
#include "Logger.h"

__global__ void MergeChannels8UKernel(int width, int height, int outputPitch, int channels, void** gpuArrayOfChannels, const size_t* gpuArrayOfChannelPitches, void* output)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementsPerThread = 4;
    const unsigned int taskSize = width * height;
    const unsigned int elementId = threadId * elementsPerThread;
    const unsigned int elementRow = elementId / width;
    const unsigned int elementColumn = elementId - (elementRow * width);

    if(elementId < taskSize)
    {
        for(unsigned int element = 0; element < elementsPerThread; ++element)
        {
            for(unsigned int channel = 0; channel < channels; ++channel)
            {
                ((unsigned char*)output)[elementRow * outputPitch + elementColumn * channels + element * channels + channel] =
                ((unsigned char**)(gpuArrayOfChannels))[channels - 1 - channel][elementRow * gpuArrayOfChannelPitches[channels - 1 - channel] + elementColumn + element];
            }
        }
    }
}

__global__ void MergeChannels16UKernel(int width, int height, int outputPitch, int channels, void** gpuArrayOfChannels, const size_t* gpuArrayOfChannelPitches, void* output)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned short elementsPerThread = 4;
    const unsigned int taskSize = width * height;
    const unsigned int elementId = threadId * elementsPerThread;
    const unsigned int elementRow = elementId / width;
    const unsigned int elementColumn = elementId - (elementRow * width);

    if(elementId < taskSize)
    {
        for(unsigned int element = 0; element < elementsPerThread; ++element)
        {
            for(unsigned int channel = 0; channel < channels; ++channel)
            {
                ((unsigned short*)output)[elementRow * outputPitch + elementColumn * channels + element * channels + channel] =
                        ((unsigned short**)(gpuArrayOfChannels))[channels - 1 - channel][elementRow * gpuArrayOfChannelPitches[channels - 1 - channel] + elementColumn + element];
            }
        }
    }
}

namespace Decoding
{

    void MergeChannels(int width, int height, int channels, int elementSize, int outputPitch, void** channelBuffers, size_t* channelPitches, void* output, void* cudaStream)
    {
        // 1 thread - 4 elements
        // 1 block - 256 threads

        const int elementsPerThread = 4;
        const int threadsPerBlock = 256;
        const int taskSize = width * height;

        dim3 blockDim(threadsPerBlock);
        dim3 gridDim( (taskSize + threadsPerBlock * elementsPerThread - 1) / (threadsPerBlock * elementsPerThread));

        void* gpuChannelsArray = nullptr;
        size_t* gpuArrayOfPitches = nullptr;

        auto status = cudaMallocAsync(&gpuChannelsArray, sizeof(void*) * channels, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to allocate memory for GPU array of channel pointers while merging channels. CUDA " \
            "error " << static_cast<int>(status) << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }

        status = cudaMallocAsync(&gpuArrayOfPitches, sizeof(size_t) * channels, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to allocate memory for GPU array of channel pitches while merging channels. CUDA " \
            "error " << static_cast<int>(status)
            << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }

        status = cudaMemcpyAsync(gpuChannelsArray, channelBuffers, sizeof(void*) * channels, cudaMemcpyKind::cudaMemcpyHostToDevice, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to copy data to GPU array of channel pitches while merging channels. CUDA " \
            "error " << static_cast<int>(status)
                        << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }

        status = cudaMemcpyAsync(gpuArrayOfPitches, channelPitches, sizeof(size_t) * channels, cudaMemcpyKind::cudaMemcpyHostToDevice, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to copy data to GPU array of channel pitches while merging channels. CUDA " \
            "error " << static_cast<int>(status)
                        << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }

        LOG_TRACE() << "Starting CUDA kernel for merging channels ...";

        if(elementSize == 8)
        {
            MergeChannels8UKernel<<<gridDim, blockDim, 256, (cudaStream_t)cudaStream>>>(
                    width, height, outputPitch, channels, (void**)gpuChannelsArray, gpuArrayOfPitches, output);
        }
        else if (elementSize == 16)
        {
            MergeChannels16UKernel<<<gridDim, blockDim, 256, (cudaStream_t)cudaStream>>>(
                    width, height, outputPitch, channels, (void**)gpuChannelsArray, gpuArrayOfPitches, output);
        }
        else
        {
            LOG_ERROR() << "Unsupported element size " << elementSize << ". Only 8 and 16 are supported for now.";
        }

        status = cudaGetLastError();
        if(status == cudaError_t::cudaSuccess)
        {
            LOG_TRACE() << "Channels were successfully merged.";
        }
        else
        {
            LOG_ERROR() << "Failed to merge NvJPEG2K image channels. CUDA error " << static_cast<int>(status)
                        << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }

        status = cudaFreeAsync(gpuArrayOfPitches, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to release GPU array of channel pitches. CUDA error " << static_cast<int>(status)
            << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }

        status = cudaFreeAsync(gpuChannelsArray, (cudaStream_t)cudaStream);
        if(status != cudaError_t::cudaSuccess)
        {
            LOG_ERROR() << "Failed to release GPU array of channel pointers. CUDA error " << static_cast<int>(status)
            << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
            throw std::runtime_error("CUDA error.");
        }
    }

}