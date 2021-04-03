/**
 * @file KafkaConsumer.h.
 *
 * @brief
 */

#ifndef KAFKA_CONSUMER_H
#define KAFKA_CONSUMER_H

#include <memory>
#include <vector>

// forward declaration for RdKafka::KafkaConsumer
namespace RdKafka
{
    class KafkaConsumer;
}

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
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
 * @class KafkaConsumer
 *
 * @brief
 */
class KafkaConsumer final
{

public:

    /**
     * @brief
     *
     * @param kafkaConfig
     */
    explicit KafkaConsumer(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    /**
     * @brief
     */
    ~KafkaConsumer();

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
     * @return
     */
    std::shared_ptr<KafkaMessage> Consume();

private:

    /**
     * @brief
     *
     * @param kafkaConfig
     */
    static void ValidateConfig(const std::shared_ptr<Config::JsonConfig>& kafkaConfig);

    ///
    int timeoutMs_ = 0;

    ///
    RdKafka::KafkaConsumer* consumer_ = nullptr;
};

}

#endif // KAFKA_CONSUMER_H
