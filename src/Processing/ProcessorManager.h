/**
 * @file ProcessorManager.h.
 *
 * @brief
 */

#ifndef PROCESSOR_MANAGER_H
#define PROCESSOR_MANAGER_H

#include <memory>
#include <unordered_map>

#include "IProcessor.h"

// forward declaration for GPU::GpuManager
namespace GPU
{
    class GpuManager;
}

// forward declaration for DataStructures::ProcessingQueueManager
namespace DataStructures
{
    class ProcessingQueueManager;
}

// forward declaration for Algorithms::IAlgorithmFactory
namespace Algorithms
{
    class IAlgorithmFactory;
}

// forward declaration for Config::JsonConfigManager
namespace Config
{
    class JsonConfigManager;
}

/**
 * @namespace Processing
 *
 * @brief
 */
namespace Processing
{

/**
 * @class ProcessorManager
 *
 * @brief
 */
class ProcessorManager
{

public:

    /**
     * @brief
     */
    ProcessorManager();

    /**
     * @brief
     */
    ~ProcessorManager();

    /**
     * @brief
     *
     * @param processorName
     * @return
     */
    [[nodiscard]] std::shared_ptr<IProcessor> GetProcessor(const std::string & processorName) const;

    /**
     * @brief
     *
     * @param serviceConfig
     * @param configManager
     * @param gpuManager
     * @param queueManager
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& serviceConfig,
                    const std::unique_ptr<Config::JsonConfigManager>& configManager,
                    const std::unique_ptr<GPU::GpuManager>& gpuManager,
                    const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    /**
     * @brief
     *
     * @param config
     * @param configManager
     * @param gpuManager
     * @param queueManager
     */
    void AddProcessor(const std::shared_ptr<Config::JsonConfig>& config,
                      const std::unique_ptr<Config::JsonConfigManager>& configManager,
                      const std::unique_ptr<GPU::GpuManager>& gpuManager,
                      const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    /**
     * @brief
     *
     * @param processorName
     */
    void RemoveProcessor(const std::string &processorName);

    /**
     * @brief
     *
     * @param processorName
     */
    void StartProcessor(const std::string &processorName);

    /**
     * @brief
     *
     * @param processorName
     */
    void StopProcessor(const std::string &processorName);

    /**
     * @brief
     */
    void StartAllProcessors();

    /**
     * @brief
     */
    void StopAllProcessors();

private:

    /**
     * @brief
     *
     * @param processorConfig
     */
    static void ValidateProcessorConfig(const std::shared_ptr<Config::JsonConfig>& processorConfig);

    ///
    std::unordered_map<std::string, std::shared_ptr<IProcessor>> processors_;

    ///
    std::unique_ptr<Algorithms::IAlgorithmFactory> algorithmFactory_;
};

}

#endif // PROCESSOR_MANAGER_H
