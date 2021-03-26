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
    explicit KafkaProducer(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    ~KafkaProducer();

    bool Initialize(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    void Produce(const std::shared_ptr<KafkaMessage>& message);
private:

    static void ValidateConfig(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    RdKafka::Producer* producer_ = nullptr;

    std::vector<std::string> topics_;

    int timeoutMs_ = 0;
};

}

#endif // KAFKA_PRODUCER_H
