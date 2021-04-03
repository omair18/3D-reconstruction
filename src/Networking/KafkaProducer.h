/**
 * @file KafkaProducer.h.
 *
 * @brief
 */

#ifndef KAFKA_PRODUCER_H
#define KAFKA_PRODUCER_H

#include <memory>
#include <vector>

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

// forward declaration for RdKafka::Producer
namespace RdKafka
{
    class Producer;
}

/**
 * @namespace Networking
 *
 * @brief
 */
namespace Networking
{

// forward declaration for Networking::KafkaMessage
class KafkaMessage;

/**
 * @class KafkaProducer
 *
 * @brief
 */
class KafkaProducer final
{
public:

    /**
     * @brief
     *
     * @param kafkaConfig
     */
    explicit KafkaProducer(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    /**
     * @brief
     */
    ~KafkaProducer();

    /**
     * @brief
     *
     * @param kafkaConfig
     * @return
     */
    bool Initialize(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    /**
     * @brief
     *
     * @param message
     */
    void Produce(const std::shared_ptr<KafkaMessage>& message);

private:

    /**
     * @brief
     *
     * @param kafkaConfig
     */
    static void ValidateConfig(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    ///
    RdKafka::Producer* producer_ = nullptr;

    ///
    std::vector<std::string> topics_;

    ///
    int timeoutMs_ = 0;
};

}

#endif // KAFKA_PRODUCER_H
