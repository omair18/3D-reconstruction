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

    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

private:
    std::unique_ptr<Networking::KafkaConsumer> consumer_;
};

}

#endif // KAFKA_CONSUMPTION_ALGORITHM_H
