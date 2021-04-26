/**
 * @file ProcessingData.h.
 *
 * @brief
 */

#ifndef PROCESSING_DATA_H
#define PROCESSING_DATA_H

#include <memory>

#include "ModelDataset.h"

// forward declaration for Networking::KafkaMessage
namespace Networking
{
    class KafkaMessage;
}

/**
 * @namespace DataStructures
 *
 * @brief
 */
namespace DataStructures
{

/**
 * @class ProcessingData
 *
 * @brief
 */
class ProcessingData final
{

public:

    /**
     * @brief
     */
    ProcessingData();

    /**
     * @brief
     *
     * @param other
     */
    ProcessingData(const ProcessingData& other);

    /**
     * @brief
     *
     * @param other
     */
    ProcessingData(ProcessingData&& other) noexcept;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    ProcessingData& operator=(const ProcessingData& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    ProcessingData& operator=(ProcessingData&& other) noexcept;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] const std::shared_ptr<ModelDataset>& GetModelDataset() const noexcept;

    /**
     * @brief
     *
     * @param dataset
     */
    void SetModelDataset(const std::shared_ptr<ModelDataset>& dataset);

    /**
     * @brief
     *
     * @param dataset
     */
    void SetModelDataset(const ModelDataset& dataset);

    /**
     * @brief
     *
     * @param dataset
     */
    void SetModelDataset(std::shared_ptr<ModelDataset>&& dataset) noexcept;

    /**
     * @brief
     *
     * @param dataset
     */
    void SetModelDataset(ModelDataset&& dataset) noexcept;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] const std::shared_ptr<Networking::KafkaMessage>& GetKafkaMessage() const;

    /**
     * @brief
     *
     * @param message
     */
    void SetKafkaMessage(const std::shared_ptr<Networking::KafkaMessage>& message);

    /**
     * @brief
     *
     * @param message
     */
    void SetKafkaMessage(std::shared_ptr<Networking::KafkaMessage>&& message) noexcept;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] const std::shared_ptr<Config::JsonConfig>& GetReconstructionParams() const;

    /**
     * @brief
     *
     * @param params
     */
    void SetReconstructionParams(const std::shared_ptr<Config::JsonConfig>& params);

    /**
     * @brief
     *
     * @param params
     */
    void SetReconstructionParams(std::shared_ptr<Config::JsonConfig>&& params) noexcept;

private:

    ///
    std::shared_ptr<Config::JsonConfig> reconstructionParams_ = nullptr;

    ///
    std::shared_ptr<Networking::KafkaMessage> kafkaMessage_ = nullptr;

    ///
    std::shared_ptr<ModelDataset> modelDataset_ = nullptr;

};

}
#endif // PROCESSING_DATA_H
