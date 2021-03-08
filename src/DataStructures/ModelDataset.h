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

private:

    ///
    std::vector<CUDAImageDescriptor> imageDescriptors_;

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
