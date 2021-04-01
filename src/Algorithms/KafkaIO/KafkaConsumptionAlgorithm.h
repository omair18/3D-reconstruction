#ifndef KAFKA_CONSUMPTION_ALGORITHM_H
#define KAFKA_CONSUMPTION_ALGORITHM_H

#include "ICPUAlgorithm.h"

namespace Networking
{
    class KafkaConsumer;
}

namespace Algorithms
{

class KafkaConsumptionAlgorithm final : public ICPUAlgorithm
{
public:

    KafkaConsumptionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                              [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                              [[maybe_unused]] void* cudaStream);

    ~KafkaConsumptionAlgorithm() override;

    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    std::unique_ptr<Networking::KafkaConsumer> consumer_;
};

}

#endif // KAFKA_CONSUMPTION_ALGORITHM_H
