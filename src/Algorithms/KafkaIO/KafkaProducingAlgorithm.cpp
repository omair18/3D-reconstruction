#include "KafkaProducingAlgorithm.h"
#include "KafkaProducer.h"
#include "KafkaMessage.h"
#include "ProcessingData.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace Algorithms
{

KafkaProducingAlgorithm::KafkaProducingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm()
{
    Initialize(config);
}

bool KafkaProducingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& message = processingData->GetKafkaMessage();
    if(message->Empty())
    {
        return false;
    }

    producer_->Produce(message);
    return true;
}

void KafkaProducingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    LOG_TRACE() << "Initializing " << Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaProducingAlgorithm
                << " ...";
    producer_ = std::make_unique<Networking::KafkaProducer>(config);
    LOG_TRACE() << Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaProducingAlgorithm
                << " was successfully initialized.";
}

}