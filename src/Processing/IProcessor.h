#ifndef INTERFACE_PROCESSOR_H
#define INTERFACE_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>

namespace DataStructures
{
    class ProcessingData;

    template <typename T>
    class ProcessingQueue;

    class ProcessingQueueManager;
}

namespace Algorithms
{
    class IAlgorithm;
    class IAlgorithmFactory;
}

namespace Config
{
    class JsonConfig;
    class JsonConfigManager;
}

namespace GPU
{
    class GpuManager;
}

namespace Processing
{

class IProcessor
{
public:
    IProcessor(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    virtual void Process() = 0;

    virtual void Stop() = 0;

    virtual void InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                                      const std::unique_ptr<Config::JsonConfigManager>& configManager,
                                      const std::unique_ptr<GPU::GpuManager>& gpuManager) = 0;

    virtual void Initialize() = 0;

    virtual bool IsStarted() = 0;

    [[nodiscard]] const std::string& GetName() const
    {
        return name_;
    }

    void SetName(const std::string& name)
    {
        name_ = name;
    };

    void SetName(std::string&& name)
    {
        name_ = std::move(name);
    };

    virtual ~IProcessor() = default;

protected:

    std::string name_;

    std::shared_ptr<DataStructures::ProcessingQueue<std::shared_ptr<DataStructures::ProcessingData>>> inputQueue_;

    std::shared_ptr<DataStructures::ProcessingQueue<std::shared_ptr<DataStructures::ProcessingData>>> outputQueue_;

    std::vector<std::unique_ptr<Algorithms::IAlgorithm>> processingAlgorithms_;
};


}

#endif // INTERFACE_PROCESSOR_H
