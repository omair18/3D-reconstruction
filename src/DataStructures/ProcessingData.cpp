#include "ProcessingData.h"
#include "KafkaMessage.h"
#include "JsonConfig.h"

namespace DataStructures
{

ProcessingData::ProcessingData() :
kafkaMessage_(nullptr),
modelDataset_(nullptr),
reconstructionParams_(nullptr)
{

}

ProcessingData::ProcessingData(const DataStructures::ProcessingData& other)
{
    kafkaMessage_ = std::make_unique<Networking::KafkaMessage>(*other.kafkaMessage_);
    modelDataset_ = std::make_unique<ModelDataset>(*other.modelDataset_);
    reconstructionParams_ = std::make_shared<Config::JsonConfig>(*other.reconstructionParams_);
}

ProcessingData::ProcessingData(ProcessingData&& other) noexcept
{
    kafkaMessage_ = std::move(other.kafkaMessage_);
    modelDataset_ = std::move(other.modelDataset_);
    reconstructionParams_ = std::move(other.reconstructionParams_);
}

ProcessingData &ProcessingData::operator=(const ProcessingData& other)
{
    if (this == &other)
    {
        return *this;
    }

    kafkaMessage_ = std::make_unique<Networking::KafkaMessage>(*other.kafkaMessage_);
    modelDataset_ = std::make_unique<ModelDataset>(*other.modelDataset_);
    reconstructionParams_ = std::make_shared<Config::JsonConfig>(*other.reconstructionParams_);
    return *this;
}

ProcessingData &ProcessingData::operator=(ProcessingData&& other) noexcept
{
    kafkaMessage_ = std::move(other.kafkaMessage_);
    modelDataset_ = std::move(other.modelDataset_);
    reconstructionParams_ = std::move(other.reconstructionParams_);
    return *this;
}

const std::shared_ptr<ModelDataset>& ProcessingData::GetModelDataset() const noexcept
{
    return modelDataset_;
}

void ProcessingData::SetModelDataset(const ModelDataset &dataset)
{
    modelDataset_ = std::make_shared<ModelDataset>(dataset);
}

void ProcessingData::SetModelDataset(ModelDataset&& dataset) noexcept
{
    *modelDataset_ = std::move(dataset);
}

void ProcessingData::SetModelDataset(const std::shared_ptr<ModelDataset>& dataset)
{

}

void ProcessingData::SetModelDataset(std::shared_ptr<ModelDataset>&& dataset) noexcept
{
    modelDataset_ = std::move(dataset);
}

const std::shared_ptr<Networking::KafkaMessage> &ProcessingData::GetKafkaMessage() const
{
    return kafkaMessage_;
}

void ProcessingData::SetKafkaMessage(const std::shared_ptr<Networking::KafkaMessage>& message)
{
    kafkaMessage_ = message;
}

void ProcessingData::SetKafkaMessage(std::shared_ptr<Networking::KafkaMessage>&& message) noexcept
{
    kafkaMessage_ = std::move(message);
}

const std::shared_ptr<Config::JsonConfig> &ProcessingData::GetReconstructionParams() const
{
    return reconstructionParams_;
}

void ProcessingData::SetReconstructionParams(const std::shared_ptr<Config::JsonConfig> &params)
{
    reconstructionParams_ = params;
}

void ProcessingData::SetReconstructionParams(std::shared_ptr<Config::JsonConfig> &&params) noexcept
{
    reconstructionParams_ = std::move(params);
}

}