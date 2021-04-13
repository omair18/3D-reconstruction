#include "AlgorithmFactory.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "KafkaIO/KafkaConsumptionAlgorithm.h"
#include "KafkaIO/KafkaProducingAlgorithm.h"
#include "KafkaIO/KafkaMessageParsingAlgorithm.h"
#include "Decoding/CpuImageDecodingAlgorithm.h"
#include "Decoding/CUDAImageDecodingAlgorithm.h"
#include "KafkaConsumer.h"
#include "KafkaProducer.h"
#include "Logger.h"

namespace Algorithms
{

AlgorithmFactory::AlgorithmFactory()
{
    algorithmLambdas_ =
    {
        { Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaConsumptionAlgorithm, GetAlgorithmLambda<KafkaConsumptionAlgorithm>() },
        { Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaProducingAlgorithm, GetAlgorithmLambda<KafkaProducingAlgorithm>() },
        { Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaMessageParsingAlgorithm, GetAlgorithmLambda<KafkaMessageParsingAlgorithm>() },
        { Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::CpuImageDecodingAlgorithm, GetAlgorithmLambda<CpuImageDecodingAlgorithm>() },
        { Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::CUDAImageDecodingAlgorithm, GetAlgorithmLambda<CUDAImageDecodingAlgorithm>() }
    };
}

std::unique_ptr<IAlgorithm> AlgorithmFactory::Create(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream)
{
    const auto algorithmName = (*config)[Config::ConfigNodes::AlgorithmsConfig::Name]->ToString();

    if (auto it = algorithmLambdas_.find(algorithmName); it != algorithmLambdas_.end())
    {
        LOG_TRACE() << "Creating " << algorithmName << " ...";
        const auto algorithmConfiguration = (*config)[Config::ConfigNodes::AlgorithmsConfig::Configuration];

        return it->second(algorithmConfiguration, gpuManager, cudaStream);
    }
    else
    {
        LOG_ERROR() << "Failed to create " << algorithmName << ". There is no algorithm with such name.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}


}