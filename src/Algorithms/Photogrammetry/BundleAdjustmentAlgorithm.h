/**
 * @file BundleAdjustmentAlgorithm.h
 *
 * @brief
 */

#ifndef BUNDLE_ADJUSTMENT_ALGORITHM_H
#define BUNDLE_ADJUSTMENT_ALGORITHM_H

#include "ICPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class BundleAdjustmentAlgorithm
 *
 * @brief
 */
class BundleAdjustmentAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    BundleAdjustmentAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~BundleAdjustmentAlgorithm() override = default;

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
     * @param resectionMethod
     * @return
     */
    static bool ValidateResectionMethod(const std::string& resectionMethod);

    /**
     * @brief
     *
     * @param triangulationMethod
     * @return
     */
    static bool ValidateTriangulationMethod(const std::string& triangulationMethod);

    /**
     * @brief
     *
     * @param config
     */
    void InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config);

    ///
    bool isInitialized_;

    ///
    bool saveResult_;

    ///
    bool useMotionPrior_;

    ///
    bool constantFocalLength_;

    ///
    bool constantPrincipalPoint_;

    ///
    bool constantDistortionParams_;

    ///
    std::string triangulationMethod_;

    ///
    std::string resectionMethod_;

};

}

#endif // BUNDLE_ADJUSTMENT_ALGORITHM_H
