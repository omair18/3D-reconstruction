#ifndef KAFKA_CONSUMPTION_ALGORITHM_H
#define KAFKA_CONSUMPTION_ALGORITHM_H

#include "IAlgorithm.h"

namespace Networking
{
    class KafkaConsumer;
}

namespace Algorithms
{

class KafkaConsumptionAlgorithm : public IAlgorithm
{
public:

private:
    std::unique_ptr<Networking::KafkaConsumer> consumer_;
};

}

#endif // KAFKA_CONSUMPTION_ALGORITHM_H
