#include "ProcessingData.h"

namespace DataStructures
{

ProcessingData::ProcessingData()
{

}

ProcessingData::ProcessingData(const DataStructures::ProcessingData& other)
{

}

ProcessingData::ProcessingData(ProcessingData&& other) noexcept
{

}

ProcessingData &ProcessingData::operator=(const ProcessingData& other)
{
    return *this;
}

ProcessingData &ProcessingData::operator=(ProcessingData&& other) noexcept
{
    return *this;
}

const ModelDataset &ProcessingData::GetModelDataset()
{
    return modelDataset_;
}

void ProcessingData::SetModelDataset(const ModelDataset &dataset)
{
    modelDataset_ = dataset;
}

void ProcessingData::SetModelDataset(ModelDataset &&dataset) noexcept
{
    modelDataset_ = std::move(dataset);
}

const std::shared_ptr<Networking::KafkaMessage> &ProcessingData::GetKafkaMessage()
{
    return kafkaMessage_;
}

void ProcessingData::SetKafkaMessage(const std::shared_ptr<Networking::KafkaMessage> &message)
{
    kafkaMessage_ = message;
}

void ProcessingData::SetKafkaMessage(std::shared_ptr<Networking::KafkaMessage> &&message) noexcept
{
    kafkaMessage_ = std::move(message);
}

const std::shared_ptr<Config::JsonConfig> &ProcessingData::GetReconstructionParams()
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

CUDAImage &ProcessingData::GetDecodedImage()
{
    return decodedImage_;
}

void ProcessingData::SetDecodedImage(const CUDAImage &image)
{
    decodedImage_.CopyFromCUDAImage(image);
}

void ProcessingData::SetDecodedImage(CUDAImage &&image) noexcept
{
    decodedImage_.MoveFromCUDAImage(image);
}

void ProcessingData::SetDecodedImageAsync(const CUDAImage &image, void *cudaStream)
{
    decodedImage_.CopyFromCUDAImageAsync(image, cudaStream);
}

void ProcessingData::SetDecodedImageAsync(CUDAImage &&image, void *cudaStream) noexcept
{
    decodedImage_.MoveFromCUDAImageAsync(image, cudaStream);
}


}