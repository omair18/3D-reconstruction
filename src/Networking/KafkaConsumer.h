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
    explicit KafkaConsumer(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    ~KafkaConsumer();

    bool Initialize(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    std::shared_ptr<KafkaMessage> Consume();
private:

    static void ValidateConfig(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    int timeoutMs_ = 0;
    RdKafka::KafkaConsumer* consumer_ = nullptr;
};

}

#endif // KAFKA_CONSUMER_H
