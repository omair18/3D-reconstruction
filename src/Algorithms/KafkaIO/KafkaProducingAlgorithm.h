#ifndef KAFKA_PRODUCING_ALGORITHM_H
#define KAFKA_PRODUCING_ALGORITHM_H

#include "IAlgorithm.h"

namespace Networking
{
    class KafkaProducer;
}

namespace Algorithms
{

class KafkaProducingAlgorithm : public IAlgorithm
{

public:

private:
    std::unique_ptr<Networking::KafkaProducer> producer_;
};

}

#endif // KAFKA_PRODUCING_ALGORITHM_H
