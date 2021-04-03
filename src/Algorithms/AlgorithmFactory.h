/**
 * @file AlgorithmFactory.h.
 *
 * @brief
 */

#ifndef ALGORITHM_FACTORY_H
#define ALGORITHM_FACTORY_H

#include <functional>
#include <unordered_map>

#include "IAlgorithmFactory.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class AlgorithmFactory
 *
 * @brief
 */
class AlgorithmFactory final: public IAlgorithmFactory
{

public:

    /**
     * @brief
     */
    AlgorithmFactory();

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     * @return
     */
    std::unique_ptr<IAlgorithm> Create(const std::shared_ptr<Config::JsonConfig>& config,
                                       const std::unique_ptr<GPU::GpuManager>& gpuManager,
                                       void* cudaStream) override;

    /**
     * @brief
     */
    ~AlgorithmFactory() override = default;

private:

    /**
     * @brief
     *
     * @tparam T
     * @return
     */
    template<typename T>
    auto GetAlgorithmLambda()
    {
        return [](const auto & config, const auto & gpuManager, void* cudaStream) -> std::unique_ptr<IAlgorithm>
        {
            return std::make_unique<T>(config, gpuManager, cudaStream);
        };
    }

    ///
    std::unordered_map<std::string, std::function<std::unique_ptr<IAlgorithm>(const std::shared_ptr<Config::JsonConfig>&,
                                                                              const std::unique_ptr<GPU::GpuManager>&,
                                                                              void*)>> algorithmLambdas_;
};

}

#endif // ALGORITHM_FACTORY_H
