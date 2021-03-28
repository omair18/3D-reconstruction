#ifndef GPU_PROCESSOR_H
#define GPU_PROCESSOR_H

#include <driver_types.h>

#include "IProcessor.h"
#include "EndlessThread.h"

namespace Processing
{

class GpuProcessor final : public IProcessor
{

public:

    GpuProcessor(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    ~GpuProcessor() override;

    void Process() override;

    void Stop() override;

    bool IsStarted() override;

    void Initialize() override;

    void InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                              const std::unique_ptr<Config::JsonConfigManager>& configManager,
                              const std::unique_ptr<GPU::GpuManager>& gpuManager) override;

private:

    cudaStream_t cudaStream_ = nullptr;

    EndlessThread thread_;

};

}


#endif // GPU_PROCESSOR_H
