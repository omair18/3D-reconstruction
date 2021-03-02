#ifndef MODEL_DATASET_H
#define MODEL_DATASET_H

#include <vector>
#include <string>

#include "CUDAImageDescriptor.h"

namespace Config
{
    class JsonConfig;
}

namespace DataStructures
{

class ModelDataset final
{

public:
    enum ProcessingState
    {
        COLLECTING = 0,
        PROCESSING,
        FAILED,
        READY
    };

private:
    std::vector<CUDAImageDescriptor> imageDescriptors_;
    std::string UUID_;
    int total_;
    int totalSize_;
    ProcessingState state_;
};

}

#endif //MODEL_DATASET_H
