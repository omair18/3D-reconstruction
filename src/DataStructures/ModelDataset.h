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
        COLLECTING = 0,

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
    ModelDataset() = default;

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
    ModelDataset& operator=(const ModelDataset& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    ModelDataset& operator=(ModelDataset&& other);

    /**
     * @brief
     *
     * @return
     */
    const std::string& GetUUID();

    /**
     * @brief
     *
     * @param UUID
     */
    void SetUUID(const std::string& UUID);

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

private:

    ///
    std::vector<CUDAImageDescriptor> imagesDescriptors_;

    ///
    std::string UUID_;

    ///
    int total_;

    ///
    int totalSize_;

    ///
    ProcessingStatus status_;
};

}

#endif //MODEL_DATASET_H
