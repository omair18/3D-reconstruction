#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <memory>

namespace Config
{
    class JsonConfig;
}

namespace Networking
{

class KafkaMessage;

class KafkaProducer final
{
public:
    explicit KafkaProducer(const std::shared_ptr<Config::JsonConfig>& config);

    ~KafkaProducer();

    void Produce(const std::shared_ptr<KafkaMessage>& message);
private:
    RdKafka::Producer* producer_ = nullptr;

    std::string topic_;
};

}

#endif // KAFKA_PRODUCER_H
