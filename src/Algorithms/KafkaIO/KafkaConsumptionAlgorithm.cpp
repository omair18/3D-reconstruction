#include "KafkaConsumptionAlgorithm.h"
#include "KafkaConsumer.h"
#include "KafkaMessage.h"
#include "ProcessingData.h"

namespace Algorithms
{

bool KafkaConsumptionAlgorithm::Process(std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    auto message = consumer_->Consume();
    if(message->Empty())
    {
        return false;
    }
    processingData->SetKafkaMessage(std::move(message));
    return true;
}

}