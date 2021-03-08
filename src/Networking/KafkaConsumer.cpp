#include <librdkafka/rdkafkacpp.h>

#include "KafkaConsumer.h"
#include "KafkaMessage.h"
#include "Logger.h"

Networking::KafkaConsumer::KafkaConsumer(const std::shared_ptr<Config::JsonConfig> &config)
{

}

std::shared_ptr<Networking::KafkaMessage> Networking::KafkaConsumer::Consume()
{
    RdKafka::Message* message = consumer_->consume(timeoutMs_);
/*
    switch (message->err())
    {
        case RdKafka::ERR__TIMED_OUT:
            //LOG_ERROR("Consuming failed: Time out. Error message: %s", message->errstr());
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

            }
            else if(!message->key_len())
            {

            }
            else
            {
                return std::make_shared<KafkaMessage>(message);
                delete message;
            }

            if(!receivedFrame.empty())
            {
                auto data = std::make_shared<ProcessingData>();
                data->SetImage(receivedFrame);
                data->SetKey(*message->key());

                m_queue->Put(data);
            }
            else
            {
                LOG_ERROR("Failed to read frame from message with timestamp: %s %i, offset: %i, key: %s", tsname, ts.timestamp, message->offset(), message->key());
            }

        } break;

        case RdKafka::ERR__PARTITION_EOF:
            LOG_TRACE("EOF reached while consuming message");
            break;

        case RdKafka::ERR__UNKNOWN_TOPIC:
            LOG_ERROR("Consuming failed: Unknown topic. Error message: %s", message->errstr());
            break;

        case RdKafka::ERR__UNKNOWN_PARTITION:
            LOG_ERROR("Consuming failed: Unknown partition. Error message: %s", message->errstr());
            break;

        default:
            LOG_ERROR("Consuming failed: %s", message->errstr());
    }
*/
    delete message;

}

Networking::KafkaConsumer::~KafkaConsumer()
{
    if(consumer_)
    {
        delete consumer_;
    }
}