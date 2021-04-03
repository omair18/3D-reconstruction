/**
 * @file DatasetCollectingAlgorithm.h.
 *
 * @brief
 */

#ifndef DATASET_COLLECTING_ALGORITHM_H
#define DATASET_COLLECTING_ALGORITHM_H

#include "ICPUAlgorithm.h"

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

/**
 * @namespace Algorithms
 *
 * @brief
 */
namespace Algorithms
{

/**
 * @class DatasetCollectingAlgorithm
 *
 * @brief
 */
class DatasetCollectingAlgorithm : public ICPUAlgorithm
{

public:

    /**
     * @brief
     *
     * @param config
     */
    explicit DatasetCollectingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config);

private:

};

}
#endif // DATASET_COLLECTING_ALGORITHM_H
