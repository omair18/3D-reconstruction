/**
 * @file DeviceToHostTransferringAlgorithm.h
 *
 * @brief
 */

#ifndef DEVICE_TO_HOST_TRANSFERRING_ALGORITHM_H
#define DEVICE_TO_HOST_TRANSFERRING_ALGORITHM_H

#include "IGPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class DeviceToHostTransferringAlgorithm
 *
 * @brief
 */
class DeviceToHostTransferringAlgorithm : public IGPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    DeviceToHostTransferringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream);

    /**
     * @brief
     */
    ~DeviceToHostTransferringAlgorithm() override = default;

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


};

}

#endif // DEVICE_TO_HOST_TRANSFERRING_ALGORITHM_H
