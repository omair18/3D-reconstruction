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
     * @brief
     */
    ~DatasetCollectingAlgorithm() override;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    /**
     * @brief
     *
     * @param config
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    std::unordered_map<std::string, std::pair<unsigned long, const std::shared_ptr<DataStructures::ProcessingData>>> datasets_;
};

}
#endif // DATASET_COLLECTING_ALGORITHM_H
