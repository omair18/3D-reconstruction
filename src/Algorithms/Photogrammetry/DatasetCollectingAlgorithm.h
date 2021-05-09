/**
 * @file DatasetCollectingAlgorithm.h.
 *
 * @brief
 */

#ifndef DATASET_COLLECTING_ALGORITHM_H
#define DATASET_COLLECTING_ALGORITHM_H

#include <unordered_map>

#include "ICPUAlgorithm.h"

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class DatasetCollectingAlgorithm
 *
 * @brief
 */
class DatasetCollectingAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    DatasetCollectingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief Default destructor.
     */
    ~DatasetCollectingAlgorithm() override = default;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(const std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    /**
     * @brief
     *
     * @param config
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    /**
     * @brief
     *
     * @param config
     */
    static void ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config);

    /**
     * @brief
     *
     * @param config
     */
    void InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config);

    ///
    std::unordered_map<std::string, std::pair<unsigned long, const std::shared_ptr<DataStructures::ProcessingData>>> datasets_;

    unsigned long expireTimeoutSeconds_;
};

}
#endif // DATASET_COLLECTING_ALGORITHM_H
