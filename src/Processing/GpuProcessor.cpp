#include <cuda_runtime.h>
#include <thread>

#include "GpuProcessor.h"
#include "ProcessingQueue.h"
#include "Logger.h"

Processing::GpuProcessor::GpuProcessor(const std::shared_ptr<Config::JsonConfig> &config) :
IProcessor(config)
{
    cudaError_t status;
    status = cudaStreamCreate(&cudaStream_);
    if(status != cudaError::cudaSuccess)
    {
        LOG_ERROR() << "Failed to create CUDA stream for GPU processor " << name_ <<". Details: "
        << status << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
    }
}

void Processing::GpuProcessor::Process()
{
    LOG_TRACE() << "Starting GPU processor " << name_ << " ...";
    std::thread([&]()
    {
        try
        {
            std::shared_ptr<DataStructures::ProcessingData> processingData = nullptr;
            if (inputQueue_)
            {
                inputQueue_->Get(processingData);
            }
            else
            {

            }

            for (auto& algorithm : processingAlgorithms_)
            {

            }

        }
        catch (std::exception& exception)
        {
            LOG_ERROR() << "Received exception on GPU processor " << name_ <<". " << exception.what();
        }
    }).detach();
}

Processing::GpuProcessor::~GpuProcessor()
{
    if(cudaStream_)
    {
        cudaStreamDestroy(cudaStream_);
    }
}


