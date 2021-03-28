#ifndef ALGORITHM_FACTORY_H
#define ALGORITHM_FACTORY_H

#include <functional>
#include <unordered_map>

#include "IAlgorithmFactory.h"

namespace Algorithms
{

class AlgorithmFactory final: public IAlgorithmFactory
{


public:
    explicit AlgorithmFactory();

    std::unique_ptr<IAlgorithm> Create(const std::shared_ptr<Config::JsonConfig>& config) override;

    ~AlgorithmFactory() override = default;

private:
    template<typename T>
    auto GetAlgorithmLambda();

    std::unordered_map<std::string, std::function<std::unique_ptr<IAlgorithm>(const std::shared_ptr<Config::JsonConfig>&)>> m_algorithmLambdas;
};

template<typename T>
auto AlgorithmFactory::GetAlgorithmLambda()
{
    return [](const auto & config, const auto & modelManager, const auto& interprocessObjectManager) -> std::unique_ptr<IAlgorithm>
    {
        return std::make_unique<T>(config, modelManager, interprocessObjectManager);
    };
}

}

#endif // ALGORITHM_FACTORY_H
