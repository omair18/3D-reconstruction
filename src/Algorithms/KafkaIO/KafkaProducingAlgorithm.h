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

    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

private:
    std::unique_ptr<Networking::KafkaProducer> producer_;
};

}

#endif // KAFKA_PRODUCING_ALGORITHM_H
