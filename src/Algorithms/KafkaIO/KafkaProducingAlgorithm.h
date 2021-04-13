/**
 * @file KafkaProducingAlgorithm.h.
 *
 * @brief
 */

#ifndef KAFKA_PRODUCING_ALGORITHM_H
#define KAFKA_PRODUCING_ALGORITHM_H

#include "ICPUAlgorithm.h"

// forward declaration for Networking::KafkaProducer
namespace Networking
{
    class KafkaProducer;
}

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class KafkaProducingAlgorithm
 *
 * @brief
 */
class KafkaProducingAlgorithm final: public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    KafkaProducingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                            [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                            [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~KafkaProducingAlgorithm() override = default;

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
    std::unique_ptr<Networking::KafkaProducer> producer_;
};

}

#endif // KAFKA_PRODUCING_ALGORITHM_H
