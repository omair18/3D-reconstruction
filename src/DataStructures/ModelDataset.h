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

    const std::string& GetUUID();

    void SetUUID(const std::string& UUID);

    std::string GetProcessingStatusString();

    ProcessingStatus GetProcessingStatus();

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
