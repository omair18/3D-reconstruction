#include <librdkafka/rdkafkacpp.h>
#include <boost/algorithm/string/trim.hpp>

#include "KafkaConsumer.h"
#include "KafkaMessage.h"
#include "JsonConfig.h"
#include "Logger.h"
#include "ConfigNodes.h"

Networking::KafkaConsumer::KafkaConsumer(const std::shared_ptr<Config::JsonConfig> &kafkaConfig) :
timeoutMs_(-1),
consumer_(nullptr)
{
    LOG_TRACE() << "Initializing kafka consumer ...";
    if(Initialize(kafkaConfig))
    {
        LOG_TRACE() << "Kafka consumer was initialized successfully.";
    }
    else
    {
        LOG_ERROR() << "Failed to initialize kafka consumer.";
    }
}

std::shared_ptr<Networking::KafkaMessage> Networking::KafkaConsumer::Consume()
{
    RdKafka::Message* message = consumer_->consume(timeoutMs_);
    switch (message->err())
    {
        case RdKafka::ERR__TIMED_OUT:
            //produces too much log records
            //LOG_ERROR() << "Consumption failed: Time out. Error message: " << message->errstr();
            break;

        case RdKafka::ERR_NO_ERROR:
        {
            RdKafka::MessageTimestamp ts{};
            std::string tsname = "?";
            ts = message->timestamp();
            if (ts.type != RdKafka::MessageTimestamp::MSG_TIMESTAMP_NOT_AVAILABLE)
            {
                if (ts.type == RdKafka::MessageTimestamp::MSG_TIMESTAMP_CREATE_TIME)
                    tsname = "create time";
                else if (ts.type == RdKafka::MessageTimestamp::MSG_TIMESTAMP_LOG_APPEND_TIME)
                    tsname = "log append time";
            }

            if(!message->len())
            {
                LOG_ERROR() << "Failed to read message with timestamp: " << tsname << " "
                << ts.timestamp << ", offset: " << message->offset() << ". Message's payload is empty.";
            }
            else if(!message->key_len())
            {
                LOG_ERROR() << "Failed to read message with timestamp: " << tsname << " "
                << ts.timestamp << ", offset: " << message->offset() << ". Message's key is empty.";
            }
            else
            {
                LOG_TRACE() << "Received message with key " << *message->key();
                auto kafkaMessage = std::make_shared<KafkaMessage>(message);
                delete message;
                return kafkaMessage;
            }

        } break;

        case RdKafka::ERR__PARTITION_EOF:
            LOG_ERROR() << "EOF reached while consuming message.";
            break;

        case RdKafka::ERR__UNKNOWN_TOPIC:
            LOG_ERROR() << "Consumption failed: Unknown topic. Error message: " << message->errstr();
            break;

        case RdKafka::ERR__UNKNOWN_PARTITION:
            LOG_ERROR() << "Consumption failed: Unknown partition. Error message: " << message->errstr();
            break;

        default:
            LOG_ERROR() << "Consumption failed: " << message->errstr();
    }


    delete message;

    return std::make_shared<KafkaMessage>();
}

Networking::KafkaConsumer::~KafkaConsumer()
{
    if(consumer_)
    {
        delete consumer_;
    }
}

bool Networking::KafkaConsumer::Initialize(const std::shared_ptr<Config::JsonConfig> &kafkaConfig)
{
    ValidateConfig(kafkaConfig);

    auto topics = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Topics]->ToVectorString();
    auto brokersList = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Brokers]->ToVectorString();
    bool enablePartitionEOF = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::EnablePartitionEOF]->ToBool();
    auto timeoutMs = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Timeout]->ToInt32();
    auto groupId = (*kafkaConfig)[Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::GroupId]->ToInt32();

    std::string errorString;

    RdKafka::Conf* globalConfig = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_GLOBAL);
    globalConfig->set("enable.partition.eof", enablePartitionEOF ? "true" : "false", errorString);
    if(!errorString.empty())
    {
        LOG_ERROR() << "Failed to set kafka consumer's param enable.partition.eof. Message: " << errorString;
        throw std::runtime_error("Failed to set kafka consumer's param enable.partition.eof.");
    }
    else
    {
        LOG_TRACE() << "Kafka consumer's param enable.partition.eof set to " << (enablePartitionEOF ? "true." : "false.");
    }

    globalConfig->set("group.id", std::to_string(groupId), errorString);
    if(!errorString.empty())
    {
        LOG_ERROR() << "Failed to set kafka consumer's param group.id. Message: " << errorString;
        throw std::runtime_error("Failed to set kafka consumer's param group.id.");
    }
    else
    {
        LOG_TRACE() << "Kafka consumer's param group.id set to " << groupId << ".";
    }

    RdKafka::Conf* topicConf = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_TOPIC);
    globalConfig->set("default_topic_conf", topicConf, errorString);
    delete  topicConf;

    std::string brokersListString;
    for(auto& broker : brokersList)
    {
        boost::algorithm::trim(broker);
        brokersListString += broker;
        brokersListString += ",";
    }
    brokersListString = brokersListString.substr(0, brokersListString.size() - 1);

    globalConfig->set("metadata.broker.list", brokersListString, errorString);
    if(!errorString.empty())
    {
        LOG_ERROR() << "Failed to set kafka consumer's param metadata.broker.list. Message: " << errorString;
        throw std::runtime_error("Failed to set kafka consumer's param metadata.broker.list.");
    }
    else
    {
        LOG_TRACE() << "Kafka consumer's param metadata.broker.list set to [" << brokersListString << "].";
    }

    consumer_ = RdKafka::KafkaConsumer::create(globalConfig, errorString);
    if(!errorString.empty())
    {
        LOG_ERROR() << "Failed to create kafka consumer. Message: " << errorString;
        if(consumer_)
        {
            delete consumer_;
            consumer_ = nullptr;
        }
        throw std::runtime_error("Failed to create kafka consumer.");
    }
    else
    {
        LOG_TRACE() << "Kafka consumer was successfully created.";
    }

    std::string topicsString;
    for(auto& topic : topics)
    {
        topicsString += topic;
        topicsString += ", ";
    }
    topicsString = topicsString.substr(0, topicsString.size() - 2);

    auto errorCode = consumer_->subscribe(topics);
    if(errorCode != RdKafka::ErrorCode::ERR_NO_ERROR)
    {
        LOG_ERROR() << "Failed to start listening the following topics: " << topicsString << ". "
        << "Reason : " << RdKafka::err2str(errorCode);
    }
    else
    {
        LOG_TRACE() << "Kafka consumer is listening the following topics: " << topicsString << ".";
    }

    timeoutMs_ = timeoutMs;
    LOG_TRACE() << "Kafka consumer's consumption timeout was set to " << timeoutMs << "milliseconds.";

    delete globalConfig;
    return true;
}

void Networking::KafkaConsumer::ValidateConfig(const std::shared_ptr<Config::JsonConfig> &kafkaConfig)
{
    if(!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Topics))
    {
        LOG_ERROR() << "Invalid kafka consumer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Topics
                    << " in kafka consumer configuration.";
        throw std::runtime_error("Invalid kafka consumer configuration.");
    }

    if (!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Brokers))
    {
        LOG_ERROR() << "Invalid kafka consumer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Brokers
                    << " in kafka consumer configuration.";
        throw std::runtime_error("Invalid kafka consumer configuration.");
    }

    if (!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::EnablePartitionEOF))
    {
        LOG_ERROR() << "Invalid kafka consumer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::EnablePartitionEOF
                    << " in kafka consumer configuration.";
        throw std::runtime_error("Invalid kafka consumer configuration.");
    }

    if (!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::GroupId))
    {
        LOG_ERROR() << "Invalid kafka consumer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::GroupId
                    << " in kafka consumer configuration.";
        throw std::runtime_error("Invalid kafka consumer configuration.");
    }

    if (!kafkaConfig->Contains(Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Timeout))
    {
        LOG_ERROR() << "Invalid kafka consumer configuration. There is no node "
                    << Config::ConfigNodes::NetworkingConfig::KafkaConsumerConfig::Timeout
                    << " in kafka consumer configuration.";
        throw std::runtime_error("Invalid kafka consumer configuration.");
    }
}
