/**
 * @file KafkaMessageParsingAlgorithm.h
 *
 * @brief
 */

#ifndef KAFKA_MESSAGE_PARSING_ALGORITHM_H
#define KAFKA_MESSAGE_PARSING_ALGORITHM_H

#include "ICPUAlgorithm.h"

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class KafkaMessageParsingAlgorithm
 *
 * @brief
 */
class KafkaMessageParsingAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     * @param gpuManager
     * @param cudaStream
     */
    KafkaMessageParsingAlgorithm([[maybe_unused]] const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream);

    /**
     * @brief
     */
    ~KafkaMessageParsingAlgorithm() override = default;

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

    /**
     * @brief
     *
     * @param messageKey
     * @return
     */
    static bool IsImageMessage(const std::shared_ptr<Config::JsonConfig>& messageKey);

    /**
     * @brief
     */
    void InitializeInternal();

    ///
    bool isInitialized_;
};

}

#endif //KAFKA_MESSAGE_PARSING_ALGORITHM_H
