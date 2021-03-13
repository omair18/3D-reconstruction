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

class ProcessingData
{
public:
    ProcessingData() = default;

    ProcessingData(const ProcessingData& other);

    ProcessingData(ProcessingData&& other) noexcept;

    ProcessingData& operator=(const ProcessingData& other);

    ProcessingData& operator=(ProcessingData&& other) noexcept;



private:

    std::shared_ptr<Config::JsonConfig> reconstructionParams_;

    std::shared_ptr<Networking::KafkaMessage> kafkaMessage_;

    ModelDataset modelDataset_;

};

}
#endif // PROCESSING_DATA_H
