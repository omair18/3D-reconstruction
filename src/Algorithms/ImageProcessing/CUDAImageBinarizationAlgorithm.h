/**
 * @file CUDAImageBinarizationAlgorithm.h.
 *
 * @brief
 */

#ifndef CUDA_IMAGE_BINARIZATION_ALGORITHM_H
#define CUDA_IMAGE_BINARIZATION_ALGORITHM_H

#include <vector>
#include <unordered_map>

#include "IGPUAlgorithm.h"
#include "CUDAImage.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class CUDAImageBinarizationAlgorithm
 *
 * @brief
 */
class CUDAImageBinarizationAlgorithm : public IGPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    CUDAImageBinarizationAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream);

    /**
     * @brief
     */
    ~CUDAImageBinarizationAlgorithm() override = default;

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

    /**
     * @brief
     *
     * @param width
     * @param height
     */
    void AllocateBuffer(int width, int height);


    ///
    DataStructures::CUDAImage binarizationBuffer_;

    ///
    std::unordered_map<int, std::vector<float>> binarizationCoefficients_;

    ///
    bool allowUnconfiguredChannels_;

    ///
    void* cudaStream_;

    ///
    std::shared_ptr<GPU::GPU> currentGPU_;
};

}

#endif // CUDA_IMAGE_BINARIZATION_ALGORITHM_H
