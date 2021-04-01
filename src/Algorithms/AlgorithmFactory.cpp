#include "AlgorithmFactory.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "KafkaIO/KafkaConsumptionAlgorithm.h"
#include "KafkaIO/KafkaProducingAlgorithm.h"
#include "Logger.h"

namespace Algorithms
{

AlgorithmFactory::AlgorithmFactory()
{
    algorithmLambdas_ =
    {
        { Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaConsumptionAlgorithm, GetAlgorithmLambda<KafkaConsumptionAlgorithm>() },
        { Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::KafkaProducingAlgorithm, GetAlgorithmLambda<KafkaProducingAlgorithm>() }
    };
}

std::unique_ptr<IAlgorithm> AlgorithmFactory::Create(const std::shared_ptr<Config::JsonConfig>& config,
                                                     const std::unique_ptr<GPU::GpuManager>& gpuManager,
                                                     void* cudaStream)
{
    const auto algorithmName = (*config)[Config::ConfigNodes::AlgorithmsConfig::Name]->ToString();

    if (auto it = algorithmLambdas_.find(algorithmName); it != algorithmLambdas_.end())
    {
        //LOG_INFO("Add %s algorithm", algorithmName);

        const auto algorithmConfiguration = (*config)[Config::ConfigNodes::AlgorithmsConfig::Configuration];

        return it->second(algorithmConfiguration, gpuManager, cudaStream);
    }
    else
    {
        LOG_ERROR() << "";
        throw std::runtime_error(algorithmName + " algorithm not found");
    }
}


}