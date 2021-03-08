#include "AlgorithmFactory.h"

namespace Algorithms
{

AlgorithmFactory::AlgorithmFactory(const std::shared_ptr<Config::JsonConfig>& algorithmModesConfig) :
m_algorithmModesConfig(algorithmModesConfig)
{
    m_algorithmLambdas =
    {
        //{ ConfigNodes::AlgorithmNames::FaceDetector, GetAlgorithmLambda<FaceDetectionAlgorithm>() }
    };
}

std::unique_ptr<IAlgorithm> AlgorithmFactory::Create(const std::shared_ptr<Config::JsonConfig>& algorithmConfig)
{
    /*
    const auto algorithmName = (*algorithmConfig)[CommonNames::Name]->ToString();

    if (auto it = m_algorithmLambdas.find(algorithmName); it != m_algorithmLambdas.end())
    {
        LOG_INFO("Add %s algorithm", algorithmName);

        const auto algorithmConfigurationNode = (*algorithmConfig)[ConfigNodes::ServiceConfig::Configuration];

        if ((*algorithmConfigurationNode)[ConfigNodes::ServiceConfig::Modes]->IsNull())
        {
            algorithmConfigurationNode->SetNode(ConfigNodes::ServiceConfig::Modes, m_algorithmModesConfig);
        }

        return it->second(algorithmConfigurationNode, modelManager, interprocessObjectManager);
    }
    else
    {
        throw std::runtime_error(algorithmName + " algorithm not found");
    }
     */
    return nullptr;
}


}