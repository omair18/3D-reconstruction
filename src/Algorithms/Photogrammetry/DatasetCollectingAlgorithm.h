#ifndef DATASET_COLLECTING_ALGORITHM_H
#define DATASET_COLLECTING_ALGORITHM_H

#include "ICPUAlgorithm.h"

namespace Config
{
    class JsonConfig;
}

namespace Algorithms
{

class DatasetCollectingAlgorithm : public ICPUAlgorithm
{

public:
    explicit DatasetCollectingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config);

private:

};

}
#endif // DATASET_COLLECTING_ALGORITHM_H
