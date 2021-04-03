/**
 * @file GpuProcessor.h.
 *
 * @brief
 */

#ifndef GPU_PROCESSOR_H
#define GPU_PROCESSOR_H

#include <driver_types.h>

#include "IProcessor.h"
#include "EndlessThread.h"

/**
 * @namespace Processing
 *
 * @brief
 */
namespace Processing
{

/**
 * @class GpuProcessor
 *
 * @brief
 */
class GpuProcessor final : public IProcessor
{

public:

    /**
     * @brief
     *
     * @param config
     * @param queueManager
     */
    GpuProcessor(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    /**
     * @brief
     */
    ~GpuProcessor() override;

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
     */
    void Initialize() override;

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

private:

    /**
     * @brief
     *
     * @param algorithmConfig
     */
    static void ValidateAlgorithmConfig(const std::shared_ptr<Config::JsonConfig>& algorithmConfig);

    ///
    cudaStream_t cudaStream_ = nullptr;

    ///
    EndlessThread thread_;

};

}


#endif // GPU_PROCESSOR_H
