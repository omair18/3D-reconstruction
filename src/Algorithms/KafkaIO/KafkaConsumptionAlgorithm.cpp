#include "KafkaConsumptionAlgorithm.h"
#include "KafkaConsumer.h"
#include "KafkaMessage.h"
#include "ProcessingData.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace Algorithms
{

KafkaConsumptionAlgorithm::KafkaConsumptionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm()
{
    Initialize(config);
}

bool KafkaConsumptionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto message = consumer_->Consume();
    if(message->Empty())
    {
        return false;
    }
    processingData->SetKafkaMessage(std::move(message));
    return true;
}

void KafkaConsumptionAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    LOG_TRACE() << "Initializing " << Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaConsumptionAlgorithm
    << " ...";
    consumer_ = std::make_unique<Networking::KafkaConsumer>(config);
    LOG_TRACE() << Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaConsumptionAlgorithm
                << " was successfully initialized.";
}

}