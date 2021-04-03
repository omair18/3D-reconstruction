/**
 * @file KafkaConsumptionAlgorithm.h.
 *
 * @brief
 */

#ifndef KAFKA_CONSUMPTION_ALGORITHM_H
#define KAFKA_CONSUMPTION_ALGORITHM_H

#include "ICPUAlgorithm.h"

// forward declaration for Networking::KafkaConsumer
namespace Networking
{
    class KafkaConsumer;
}

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class KafkaConsumptionAlgorithm
 *
 * @brief
 */
class KafkaConsumptionAlgorithm final : public ICPUAlgorithm
{
public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    KafkaConsumptionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                              [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                              [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~KafkaConsumptionAlgorithm() override;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    /**
     * @brief
     *
     * @param config
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    ///
    std::unique_ptr<Networking::KafkaConsumer> consumer_;
};

}

#endif // KAFKA_CONSUMPTION_ALGORITHM_H
