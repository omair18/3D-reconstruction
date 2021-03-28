#ifndef PROCESSOR_MANAGER_H
#define PROCESSOR_MANAGER_H

#include <memory>
#include <unordered_map>

#include "IProcessor.h"

namespace GPU
{
    class GpuManager;
}

namespace DataStructures
{
    class ProcessingQueueManager;
}

namespace Algorithms
{
    class IAlgorithmFactory;
}

namespace Config
{
    class JsonConfigManager;
}

namespace Processing
{

class ProcessorManager
{

public:

    ProcessorManager();

    ~ProcessorManager();

    [[nodiscard]] std::shared_ptr<IProcessor> GetProcessor(const std::string & processorName) const;

    void Initialize(const std::shared_ptr<Config::JsonConfig>& serviceConfig,
                    const std::unique_ptr<Config::JsonConfigManager>& configManager,
                    const std::unique_ptr<GPU::GpuManager>& gpuManager,
                    const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    void AddProcessor(const std::shared_ptr<Config::JsonConfig>& config,
                      const std::unique_ptr<Config::JsonConfigManager>& configManager,
                      const std::unique_ptr<GPU::GpuManager>& gpuManager,
                      const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    void RemoveProcessor(const std::string &processorName);

    void StartProcessor(const std::string &processorName);

    void StopProcessor(const std::string &processorName);

    void StartAllProcessors();

    void StopAllProcessors();

private:

    static void ValidateProcessorConfig(const std::shared_ptr<Config::JsonConfig>& processorConfig);

    std::unordered_map<std::string, std::shared_ptr<IProcessor>> processors_;

    std::unique_ptr<Algorithms::IAlgorithmFactory> algorithmFactory_;
};

}

#endif // PROCESSOR_MANAGER_H
