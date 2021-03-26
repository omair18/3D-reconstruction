#ifndef PROCESSING_DATA_H
#define PROCESSING_DATA_H

#include <memory>

#include "ModelDataset.h"

namespace Networking
{
    class KafkaMessage;
}

namespace DataStructures
{

class ProcessingData final
{
public:
    ProcessingData();

    ProcessingData(const ProcessingData& other);

    ProcessingData(ProcessingData&& other) noexcept;

    ProcessingData& operator=(const ProcessingData& other);

    ProcessingData& operator=(ProcessingData&& other) noexcept;

    const ModelDataset& GetModelDataset();

    void SetModelDataset(const ModelDataset& dataset);

    void SetModelDataset(ModelDataset&& dataset) noexcept;

    const std::shared_ptr<Networking::KafkaMessage>& GetKafkaMessage();

    void SetKafkaMessage(const std::shared_ptr<Networking::KafkaMessage>& message);

    void SetKafkaMessage(std::shared_ptr<Networking::KafkaMessage>&& message) noexcept;

    const std::shared_ptr<Config::JsonConfig>& GetReconstructionParams();

    void SetReconstructionParams(const std::shared_ptr<Config::JsonConfig>& params);

    void SetReconstructionParams(std::shared_ptr<Config::JsonConfig>&& params) noexcept;

    CUDAImage& GetDecodedImage();

    void SetDecodedImage(const CUDAImage& image);

    void SetDecodedImage(CUDAImage&& image) noexcept;

    void SetDecodedImageAsync(const CUDAImage& image, void* cudaStream);

    void SetDecodedImageAsync(CUDAImage&& image, void* cudaStream) noexcept;

private:

    std::shared_ptr<Config::JsonConfig> reconstructionParams_;

    std::shared_ptr<Networking::KafkaMessage> kafkaMessage_;

    ModelDataset modelDataset_;

    CUDAImage decodedImage_;

};

}
#endif // PROCESSING_DATA_H
