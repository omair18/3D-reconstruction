#ifndef CPU_PROCESSOR_H
#define CPU_PROCESSOR_H

#include "IProcessor.h"
#include "EndlessThread.h"

namespace Config
{
    class JsonConfigManager;
}

namespace DataStructures
{
    class ProcessingQueueManager;
}

namespace Processing
{

class CpuProcessor : public IProcessor
{
public:
    CpuProcessor(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    ~CpuProcessor() override;

    void Process() override;

    void Stop() override;

    bool IsStarted() override;

    void InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                              const std::unique_ptr<Config::JsonConfigManager>& configManager,
                              const std::unique_ptr<GPU::GpuManager>& gpuManager) override;

    void Initialize() override;

private:

    EndlessThread thread_;

};

}
#endif // CPU_PROCESSOR_H
