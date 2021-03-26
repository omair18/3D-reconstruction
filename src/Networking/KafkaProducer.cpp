#include <librdkafka/rdkafkacpp.h>
#include <boost/algorithm/string/trim.hpp>

#include "KafkaProducer.h"
#include "KafkaMessage.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "Logger.h"

Networking::KafkaProducer::KafkaProducer(const std::shared_ptr<Config::JsonConfig> &kafkaConfig) :
producer_(nullptr)
{
    LOG_TRACE() << "Initializing kafka producer ...";
    if(Initialize(kafkaConfig))
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
    for(auto& topic : topics_)
    {
        auto errorCode = producer_->produce(topic,
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
            LOG_ERROR() << "Producing message to topic " << topic << " was not successful. Details: "
            << RdKafka::err2str(errorCode);
        }

        errorCode = producer_->flush(timeoutMs_);
        if(errorCode != RdKafka::ErrorCode::ERR_NO_ERROR)
        {
            LOG_ERROR() << "Producing message to topic " << topic << " was not successful. Details: "
                        << RdKafka::err2str(errorCode);
        }

    }

}

bool Networking::KafkaProducer::Initialize(const std::shared_ptr<Config::JsonConfig> &kafkaConfig)
{
    ValidateConfig(kafkaConfig);

    auto topics = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Topics]->ToVectorString();
    auto brokersList = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Brokers]->ToVectorString();
    auto timeoutMs = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Timeout]->ToInt();
    std::string errorString;

    std::string brokersListString;
    for(auto& broker : brokersList)
    {
        boost::algorithm::trim(broker);
        brokersListString += broker;
        brokersListString += ",";
    }
    brokersListString = brokersListString.substr(0, brokersListString.size() - 1);

    RdKafka::Conf* globalConfig = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_GLOBAL);

    RdKafka::Conf* topicConfig = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_TOPIC);
    globalConfig->set("default_topic_conf", topicConfig, errorString);
    delete topicConfig;

    globalConfig->set("metadata.broker.list", brokersListString, errorString);
    if(!errorString.empty())
    {
        LOG_ERROR() << "Failed to set kafka producer param metadata.broker.list. Message: " << errorString;
        throw std::runtime_error("Failed to set kafka producer param metadata.broker.list.");
    }
    else
    {
        LOG_TRACE() << "Kafka producer param metadata.broker.list set to [" << brokersListString << "].";
    }

    producer_ = RdKafka::Producer::create(globalConfig, errorString);
    if(!errorString.empty())
    {
        LOG_ERROR() << "Failed to set kafka producer param metadata.broker.list. Message: " << errorString;
        throw std::runtime_error("Failed to set kafka producer param metadata.broker.list.");
    }
    else
    {
        LOG_TRACE() << "Kafka producer param metadata.broker.list set to [" << brokersListString << "].";
    }

    timeoutMs_ = timeoutMs;
    LOG_TRACE() << "Kafka producer's flush timeout was set to " << timeoutMs << " milliseconds";

    topics_ = topics;

    std::string topicsString;
    for(auto& topic : topics)
    {
        topicsString += topic;
        topicsString += ", ";
    }
    topicsString = topicsString.substr(0, topicsString.size() - 2);
    LOG_TRACE() << "Kafka producer's output topics were set to " << topicsString << ".";

    delete globalConfig;

    return true;
}

void Networking::KafkaProducer::ValidateConfig(const std::shared_ptr<Config::JsonConfig> &kafkaConfig)
{
    if(!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Topics))
    {
        LOG_ERROR() << "Invalid kafka producer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Topics
                    << " in kafka producer configuration.";
        throw std::runtime_error("Invalid kafka producer configuration.");
    }

    if (!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Brokers))
    {
        LOG_ERROR() << "Invalid kafka producer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Brokers
                    << " in kafka producer configuration.";
        throw std::runtime_error("Invalid kafka producer configuration.");
    }

    if (!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Timeout))
    {
        LOG_ERROR() << "Invalid kafka producer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaProducerConfig::Timeout
                    << " in kafka producer configuration.";
        throw std::runtime_error("Invalid kafka producer configuration.");
    }
}
