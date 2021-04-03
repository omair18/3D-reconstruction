/**
 * @file CpuProcessor.h.
 *
 * @brief
 */

#ifndef CPU_PROCESSOR_H
#define CPU_PROCESSOR_H

#include "IProcessor.h"
#include "EndlessThread.h"

// forward declaration for Config::JsonConfigManager
namespace Config
{
    class JsonConfigManager;
}

// forward declaration for DataStructures::ProcessingQueueManager
namespace DataStructures
{
    class ProcessingQueueManager;
}

/**
 * @namespace Processing
 *
 * @brief
 */
namespace Processing
{

/**
 * @class CpuProcessor
 *
 * @brief
 */
class CpuProcessor : public IProcessor
{

public:

    /**
     * @brief
     *
     * @param config
     * @param queueManager
     */
    CpuProcessor(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    /**
     * @brief
     */
    ~CpuProcessor() override;

    /**
     * @brief
     */
    void Process() override;

    /**
     * @brief
     */
    void Stop() override;

    /**
     * @brief
     *
     * @return
     */
    bool IsStarted() override;

    /**
     * @brief
     *
     * @param algorithmFactory
     * @param configManager
     * @param gpuManager
     */
    void InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                              const std::unique_ptr<Config::JsonConfigManager>& configManager,
                              const std::unique_ptr<GPU::GpuManager>& gpuManager) override;

    /**
     * @brief
     */
    void Initialize() override;

private:

    /**
     * @brief
     *
     * @param algorithmConfig
     */
    static void ValidateAlgorithmConfig(const std::shared_ptr<Config::JsonConfig>& algorithmConfig);

    ///
    EndlessThread thread_;

};

}
#endif // CPU_PROCESSOR_H
