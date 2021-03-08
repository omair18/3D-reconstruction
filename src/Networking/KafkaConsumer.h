#ifndef KAFKA_CONSUMER_H
#define KAFKA_CONSUMER_H

#include <memory>
#include <vector>

namespace RdKafka
{
    class KafkaConsumer;
}

namespace Config
{
    class JsonConfig;
}

namespace Networking
{

class KafkaMessage;

class KafkaConsumer final
{
public:
    explicit KafkaConsumer(const std::shared_ptr<Config::JsonConfig>& config);

    ~KafkaConsumer();

    std::shared_ptr<KafkaMessage> Consume();
private:

    int timeoutMs_ = 0;
    RdKafka::KafkaConsumer* consumer_ = nullptr;
};

}

#endif // KAFKA_CONSUMER_H
