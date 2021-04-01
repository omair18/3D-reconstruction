#ifndef KAFKA_PRODUCING_ALGORITHM_H
#define KAFKA_PRODUCING_ALGORITHM_H

#include "ICPUAlgorithm.h"

namespace Networking
{
    class KafkaProducer;
}

namespace Algorithms
{

class KafkaProducingAlgorithm final: public ICPUAlgorithm
{

public:

    KafkaProducingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                            [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                            [[maybe_unused]] void* cudaStream);

    ~KafkaProducingAlgorithm() override;

    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    std::unique_ptr<Networking::KafkaProducer> producer_;
};

}

#endif // KAFKA_PRODUCING_ALGORITHM_H
