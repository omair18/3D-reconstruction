#include <librdkafka/rdkafkacpp.h>

#include "KafkaProducer.h"
#include "KafkaMessage.h"
#include "JsonConfig.h"
#include "Logger.h"

Networking::KafkaProducer::KafkaProducer(const std::shared_ptr<Config::JsonConfig> &config)
{

}

Networking::KafkaProducer::~KafkaProducer()
{
    if (producer_)
    {
        delete producer_;
    }
}

void Networking::KafkaProducer::Produce(const std::shared_ptr<KafkaMessage> &message)
{
    std::string key = message->GetKey()->Dump();
    producer_->produce(topic_,
                        RdKafka::Topic::PARTITION_UA,
                        RdKafka::Producer::RK_MSG_COPY,
                       (void*)message->GetData().data(),
                        message->GetData().size(),
                        key.data(),
                        key.length(),
                        0,
                        nullptr,
                        nullptr);

}
