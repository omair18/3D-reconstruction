/**
 * @file HostToDeviceTransferringAlgorithm.h
 *
 * @brief
 */

#ifndef HOST_TO_DEVICE_TRANSFERRING_ALGORITHM_H
#define HOST_TO_DEVICE_TRANSFERRING_ALGORITHM_H

#include "IGPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class HostToDeviceTransferringAlgorithm
 *
 * @brief
 */
class HostToDeviceTransferringAlgorithm : public IGPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    HostToDeviceTransferringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream);

    /**
     * @brief
     */
    ~HostToDeviceTransferringAlgorithm() override = default;

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

#endif // HOST_TO_DEVICE_TRANSFERRING_ALGORITHM_H
