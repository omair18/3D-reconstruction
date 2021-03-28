#include <cuda_runtime.h>

#include "IGPUAlgorithm.h"
#include "GpuProcessor.h"
#include "ProcessingQueue.h"
#include "Logger.h"

namespace Processing
{

GpuProcessor::GpuProcessor(const std::shared_ptr<Config::JsonConfig> &config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager) :
IProcessor(config, queueManager)
{
    cudaError_t status;
    status = cudaStreamCreate(&cudaStream_);
    if(status != cudaError::cudaSuccess)
    {
        LOG_ERROR() << "Failed to create CUDA stream for GPU processor " << name_ <<". Details: "
                    << status << ": " << cudaGetErrorName(status) << " - " << cudaGetErrorString(status);
    }
}

void GpuProcessor::Process()
{
    LOG_TRACE() << "Starting GPU processor " << name_ << " ...";
}

GpuProcessor::~GpuProcessor()
{
    thread_.Destroy();
    if(cudaStream_)
    {
        cudaStreamDestroy(cudaStream_);
    }
}

void GpuProcessor::Stop()
{
    thread_.Stop();
}

void GpuProcessor::InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                                        const std::unique_ptr<Config::JsonConfigManager>& configManager,
                                        const std::unique_ptr<GPU::GpuManager>& gpuManager)
{

}

void GpuProcessor::Initialize()
{
    thread_.SetExecutableFunction([&]()
    {
        try
        {
            if (inputQueue_)
            {

            }
            else
            {

            }

        }
        catch (std::exception& exception)
        {
            LOG_ERROR() << "Received exception on GPU processor " << name_ <<". " << exception.what();
        }
    }
    );
}

bool GpuProcessor::IsStarted()
{
    return thread_.IsStarted();
}

}


