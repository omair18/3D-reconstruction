/**
 * @file IProcessor.h.
 *
 * @brief
 */

#ifndef INTERFACE_PROCESSOR_H
#define INTERFACE_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>

// forward declaration for DataStructures::ProcessingData, DataStructures::ProcessingQueue and DataStructures::ProcessingQueueManager
namespace DataStructures
{
    class ProcessingData;

    template <typename T>
    class ProcessingQueue;

    class ProcessingQueueManager;
}

// forward declaration for Algorithms::IAlgorithm and Algorithms::IAlgorithmFactory
namespace Algorithms
{
    class IAlgorithm;
    class IAlgorithmFactory;
}

// forward declaration for Config::JsonConfig and Config::JsonConfigManager
namespace Config
{
    class JsonConfig;
    class JsonConfigManager;
}

// forward declaration for GPU::GpuManager
namespace GPU
{
    class GpuManager;
}

/**
 * @namespace Processing
 *
 * @brief
 */
namespace Processing
{

/**
 * @class IProcessor
 *
 * @brief
 */
class IProcessor
{

public:

    /**
     * @brief
     *
     * @param config
     * @param queueManager
     */
    IProcessor(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<DataStructures::ProcessingQueueManager>& queueManager);

    /**
     * @brief
     */
    virtual ~IProcessor() = default;

    /**
     * @brief
     */
    virtual void Process() = 0;

    /**
     * @brief
     */
    virtual void Stop() = 0;

    /**
     * @brief
     *
     * @param algorithmFactory
     * @param configManager
     * @param gpuManager
     */
    virtual void InitializeAlgorithms(const std::unique_ptr<Algorithms::IAlgorithmFactory>& algorithmFactory,
                                      const std::unique_ptr<Config::JsonConfigManager>& configManager,
                                      const std::unique_ptr<GPU::GpuManager>& gpuManager) = 0;

    /**
     * @brief
     */
    virtual void Initialize() = 0;

    /**
     * @brief
     *
     * @return
     */
    virtual bool IsStarted() = 0;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] const std::string& GetName() const
    {
        return name_;
    }

    /**
     * @brief
     *
     * @param name
     */
    void SetName(const std::string& name)
    {
        name_ = name;
    };

    /**
     * @brief
     *
     * @param name
     */
    void SetName(std::string&& name)
    {
        name_ = std::move(name);
    };

protected:

    ///
    std::string name_;

    ///
    std::shared_ptr<DataStructures::ProcessingQueue<std::shared_ptr<DataStructures::ProcessingData>>> inputQueue_;

    ///
    std::shared_ptr<DataStructures::ProcessingQueue<std::shared_ptr<DataStructures::ProcessingData>>> outputQueue_;

    ///
    std::vector<std::unique_ptr<Algorithms::IAlgorithm>> processingAlgorithms_;
};


}

#endif // INTERFACE_PROCESSOR_H
