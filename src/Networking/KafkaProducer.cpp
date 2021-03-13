#include <librdkafka/rdkafkacpp.h>

#include "KafkaProducer.h"
#include "KafkaMessage.h"
#include "JsonConfig.h"
#include "Logger.h"

Networking::KafkaProducer::KafkaProducer(const std::shared_ptr<Config::JsonConfig> &config) :
producer_(nullptr)
{
    LOG_TRACE() << "Initializing kafka producer ...";
    if(Initialize(config))
    {
        LOG_TRACE() << "Kafka producer was initialized successfully.";
    }
    else
    {
        LOG_ERROR() << "Failed to initialize kafka producer.";
    }
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
    auto errorCode = producer_->produce(topic_,
                        RdKafka::Topic::PARTITION_UA,
                        RdKafka::Producer::RK_MSG_COPY,
                       (void*)message->GetData().data(),
                        message->GetData().size(),
                        key.data(),
                        key.length(),
                        0,
                        nullptr,
                        nullptr);

    if(errorCode != RdKafka::ErrorCode::ERR_NO_ERROR)
    {
        LOG_ERROR() << "";

    }
    else
    {
        LOG_TRACE() << "";
    }

}

bool Networking::KafkaProducer::Initialize(const std::shared_ptr<Config::JsonConfig> &config)
{
    return false;
}
