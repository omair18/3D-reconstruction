/**
 * @file DatasetInitializationAlgorithm.h
 *
 * @brief
 */
#ifndef DATASET_INITIALIZATION_ALGORITHM_H
#define DATASET_INITIALIZATION_ALGORITHM_H

#include "ICPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class DatasetInitializationAlgorithm
 *
 * @brief
 */
class DatasetInitializationAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    DatasetInitializationAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~DatasetInitializationAlgorithm() override = default;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    bool Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData) override;

    /**
     * @brief
     *
     * @param config
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    /**
     * @brief
     */
    void InitializeInternal();

    ///
    bool isInitialized_;

};


}

#endif // DATASET_INITIALIZATION_ALGORITHM_H
