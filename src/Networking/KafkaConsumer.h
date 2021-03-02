#ifndef KAFKA_CONSUMER_H
#define KAFKA_CONSUMER_H

namespace RdKafka
{
    class KafkaConsumer;
    class Conf;
}

class KafkaConsumer
{
public:

private:
    RdKafka::KafkaConsumer* consumer_ = nullptr;

    RdKafka::Conf* consumerConfig_ = nullptr;
};


#endif // KAFKA_CONSUMER_H
