#include "ICPUAlgorithm.h"
#include "CpuProcessor.h"
#include "ProcessingData.h"
#include "ProcessingQueue.h"
#include "Logger.h"

namespace Processing
{

CpuProcessor::CpuProcessor(const std::shared_ptr<Config::JsonConfig> &config, const std::unique_ptr<DataStructures::ProcessingQueueManager> &queueManager) :
IProcessor(config, queueManager)
{

}

CpuProcessor::~CpuProcessor()
{
    thread_.Destroy();
}

void CpuProcessor::Process()
{
    LOG_TRACE() << "Starting CPU processor " << name_ << " ...";
    thread_.Start();
}

void CpuProcessor::Initialize()
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
            LOG_ERROR() << "Received exception on CPU processor " << name_ <<". " << exception.what();
        }
    });
}

void Processing::CpuProcessor::InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                                                    const std::unique_ptr<Config::JsonConfigManager>& configManager,
                                                    const std::unique_ptr<GPU::GpuManager>& gpuManager)
{

}

void Processing::CpuProcessor::Stop()
{
    thread_.Stop();
}

bool CpuProcessor::IsStarted()
{
    return thread_.IsStarted();
}

}