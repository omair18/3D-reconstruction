/**
 * @file KafkaMessage.h.
 *
 * @brief
 */

#ifndef KAFKA_MESSAGE_H
#define KAFKA_MESSAGE_H

#include <vector>
#include <memory>

// forward declaration for RdKafka::Message
namespace RdKafka
{
    class Message;
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

/**
 * @class KafkaMessage
 *
 * @brief
 */
class KafkaMessage final
{

public:

    /**
     * @brief
     */
    KafkaMessage() = default;

    /**
     * @brief
     *
     * @param message
     */
    explicit KafkaMessage(RdKafka::Message* message);

    /**
     * @brief
     *
     * @param otherMessage
     */
    KafkaMessage(const KafkaMessage& otherMessage);

    /**
     * @brief
     *
     * @param otherMessage
     */
    KafkaMessage(KafkaMessage&& otherMessage) noexcept;

    /**
     * @brief
     */
    ~KafkaMessage() = default;

    /**
     * @brief
     *
     * @param message
     * @return
     */
    KafkaMessage& operator=(const KafkaMessage& message);

    /**
     * @brief
     *
     * @param message
     * @return
     */
    KafkaMessage& operator=(KafkaMessage&& message) noexcept;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    bool operator==(const KafkaMessage& other) const;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] bool Empty() const;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] const std::vector<unsigned char>& GetData() const;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] const std::shared_ptr<Config::JsonConfig>& GetKey() const;

    /**
     * @brief
     *
     * @param data
     */
    void SetData(const std::vector<unsigned char>& data);

    /**
     * @brief
     *
     * @param key
     */
    void SetKey(const std::shared_ptr<Config::JsonConfig>& key);

private:

    ///
    std::vector<unsigned char> messageData_;

    ///
    std::shared_ptr<Config::JsonConfig> messageKey_;
};

}
#endif // KAFKA_MESSAGE_H
