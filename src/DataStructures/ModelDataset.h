/**
 * @file ModelDataset.h.
 *
 * @brief
 */

#ifndef MODEL_DATASET_H
#define MODEL_DATASET_H

#include <vector>
#include <string>

#include "CUDAImageDescriptor.h"

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

/**
 * @namespace DataStructures
 *
 * @brief
 */
namespace DataStructures
{

/**
 * @class
 *
 * @brief
 */
class ModelDataset final
{

public:

    /**
     * @enum
     *
     * @brief
     */
    enum ProcessingStatus
    {
        ///
        RECEIVED = 0,

        ///
        COLLECTING,

        ///
        PROCESSING,

        ///
        FAILED,

        ///
        READY
    };

    /**
     * @brief
     */
    ModelDataset();

    /**
     * @brief
     *
     * @param other
     */
    ModelDataset(const ModelDataset& other) = default;

    /**
     * @brief
     *
     * @param other
     */
    ModelDataset(ModelDataset&& other) noexcept;

    /**
     * @brief
     */
    ~ModelDataset() = default;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    ModelDataset& operator=(const ModelDataset& other) = default;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    ModelDataset& operator=(ModelDataset&& other) noexcept;

    /**
     * @brief
     *
     * @return
     */
    const std::string& GetUUID() noexcept;

    /**
     * @brief
     *
     * @param UUID
     */
    void SetUUID(const std::string& UUID);

    /**
     * @brief
     *
     * @param UUID
     */
    void SetUUID(std::string&& UUID);

    /**
     * @brief
     *
     * @param status
     */
    void SetProcessingStatus(ProcessingStatus status) noexcept;

    /**
     * @brief
     *
     * @return
     */
    std::string GetProcessingStatusString();

    /**
     * @brief
     *
     * @return
     */
    ProcessingStatus GetProcessingStatus();

    /**
     * @brief
     *
     * @return
     */
    const std::vector<CUDAImageDescriptor>& GetImagesDescriptors() noexcept;

    /**
     * @brief
     *
     * @param imagesDescriptors
     */
    void SetImagesDescriptors(const std::vector<CUDAImageDescriptor>& imagesDescriptors);

    /**
     * @brief
     *
     * @param imagesDescriptors
     */
    void SetImagesDescriptors(std::vector<CUDAImageDescriptor>&& imagesDescriptors) noexcept;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] int GetTotalFramesAmount() const noexcept;

    /**
     * @brief
     *
     * @param totalFramesAmount
     */
    void SetTotalFramesAmount(int totalFramesAmount) noexcept;

    /**
     * @brief
     *
     * @return
     */
    [[nodiscard]] int GetTotalSize() const noexcept;

    /**
     * @brief
     *
     * @param totalSize
     */
    void SetTotalSize(int totalSize);

private:

    ///
    std::vector<CUDAImageDescriptor> imagesDescriptors_;

    ///
    std::string UUID_;

    ///
    int totalFramesAmount_;

    ///
    int totalSize_;

    ///
    ProcessingStatus status_;
};

}

#endif //MODEL_DATASET_H
