#ifndef KAFKA_MESSAGE_H
#define KAFKA_MESSAGE_H

#include <vector>
#include <memory>

namespace RdKafka
{
    class Message;
}

namespace Config
{
    class JsonConfig;
}

namespace Networking
{

class KafkaMessage final
{

public:
    KafkaMessage() = default;

    explicit KafkaMessage(RdKafka::Message* message);

    KafkaMessage(const KafkaMessage& otherMessage);

    KafkaMessage(KafkaMessage&& otherMessage) noexcept;

    ~KafkaMessage() = default;

    KafkaMessage& operator=(const KafkaMessage& message);

    KafkaMessage& operator=(KafkaMessage&& message) noexcept;

    bool operator==(const KafkaMessage& other);

    bool Empty();

    const std::vector<unsigned char>& GetData();

    const std::shared_ptr<Config::JsonConfig>& GetKey();

    void SetData(const std::vector<unsigned char>& data);

    void SetKey(const std::shared_ptr<Config::JsonConfig>& key);

private:

    std::vector<unsigned char> messageData_;

    std::shared_ptr<Config::JsonConfig> messageKey_;
};

}
#endif // KAFKA_MESSAGE_H
