#include <librdkafka/rdkafkacpp.h>

#include "KafkaMessage.h"
#include "JsonConfig.h"

Networking::KafkaMessage::KafkaMessage(RdKafka::Message *message) :
messageData_(message->len()),
messageKey_(std::make_shared<Config::JsonConfig>())
{
    std::memcpy(messageData_.data(), message->payload(), message->len());
    messageKey_->FromJsonString(*message->key());
}

Networking::KafkaMessage::KafkaMessage(const Networking::KafkaMessage &otherMessage)
{
    messageData_ = otherMessage.messageData_;
    messageKey_ = std::make_shared<Config::JsonConfig>(*otherMessage.messageKey_);
}

Networking::KafkaMessage::KafkaMessage(Networking::KafkaMessage &&otherMessage) noexcept
{
    messageKey_ = std::move(otherMessage.messageKey_);
    messageData_ = std::move(otherMessage.messageData_);
}

bool Networking::KafkaMessage::Empty()
{
    return messageData_.empty();
}

const std::vector<unsigned char> &Networking::KafkaMessage::GetData()
{
    return messageData_;
}

const std::shared_ptr<Config::JsonConfig> &Networking::KafkaMessage::GetKey()
{
    return messageKey_;
}

void Networking::KafkaMessage::SetData(const std::vector<unsigned char> &data)
{
    messageData_ = data;
}

void Networking::KafkaMessage::SetKey(const std::shared_ptr<Config::JsonConfig> &key)
{
    messageKey_ = key;
}
