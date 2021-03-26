#include <thread>

#include "CpuProcessor.h"
#include "ProcessingData.h"
#include "ProcessingQueue.h"
#include "Logger.h"

Processing::CpuProcessor::CpuProcessor(const std::shared_ptr<Config::JsonConfig> &config) : IProcessor(config)
{

}

Processing::CpuProcessor::~CpuProcessor()
{

}

void Processing::CpuProcessor::Process()
{
    LOG_TRACE() << "Starting CPU processor " << name_ << " ...";
    std::thread([&]()
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
    }).detach();

}

void Processing::CpuProcessor::Initialize()
{

}
