#include "KafkaProducingAlgorithm.h"
#include "KafkaProducer.h"
#include "KafkaMessage.h"
#include "ProcessingData.h"

namespace Algorithms
{

bool KafkaProducingAlgorithm::Process(std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    auto& message = processingData->GetKafkaMessage();
    if(message->Empty())
    {
        return false;
    }

    producer_->Produce(message);
    return true;
}

}