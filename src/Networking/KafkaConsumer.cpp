#include <librdkafka/rdkafkacpp.h>


#include "KafkaConsumer.h"
#include "KafkaMessage.h"
#include "Logger.h"

Networking::KafkaConsumer::KafkaConsumer(const std::shared_ptr<Config::JsonConfig> &config) :
timeoutMs_(-1),
consumer_(nullptr)
{
    LOG_TRACE() << "Initializing kafka consumer ...";
    if(Initialize(config))
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
                LOG_TRACE() << "";
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

bool Networking::KafkaConsumer::Initialize(const std::shared_ptr<Config::JsonConfig> &config)
{
    const auto kafkaConfig = (*(*config)[ConfigNodes::ServiceConfig::Configuration])[ConfigNodes::ServiceConfig::Kafka];

    const auto topicsList = (*kafkaConfig)[ConfigNodes::ServiceConfig::Topics]->ToVectorString();
    std::string brokersList = (*kafkaConfig)[ConfigNodes::ServiceConfig::Servers]->ToString();
    std::string errStr;

    m_consumerConfig = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_GLOBAL);
    m_consumerConfig->set("enable.partition.eof", "true", errStr);

    checkErrors(errStr.empty(), "Kafka consumer param enable.partition.eof set to true",
                "Failed to set kafka consumer param enable.partition.eof to true. Message: %s",
                errStr.c_str())

    m_consumerConfig->set("group.id", "0", errStr);

    checkErrors(errStr.empty(), "Kafka consumer param group.id set to 0",
                "Failed to set kafka consumer param group.id to 0. Message: %s",
                errStr.c_str())

    RdKafka::Conf* topicConf = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_TOPIC);
    m_consumerConfig->set("default_topic_conf", topicConf, errStr);
    delete  topicConf;

    m_consumerConfig->set("metadata.broker.list", brokersList, errStr);

    checkErrors(errStr.empty(), "Kafka consumer param metadata.broker.list set to %s",
                "Failed to set kafka consumer param metadata.broker.list to %s. Message: %s",
                brokersList, errStr.c_str())

    m_consumer = RdKafka::KafkaConsumer::create(m_consumerConfig, errStr);

    checkErrors(errStr.empty(), "Kafka consumer param group.id set to %s",
                "Failed to create kafka consumer",
                brokersList)

    m_consumer->subscribe(topicsList);

    delete m_consumerConfig;

    m_consumerConfig = nullptr;

}
