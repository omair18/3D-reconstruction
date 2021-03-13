/**
 * @file IAlgorithm.h.
 *
 * @brief Declares the IAlgorithm class. This is a base class for different types of algorithms.
 */

#ifndef INTERFACE_ALGORITHM_H
#define INTERFACE_ALGORITHM_H

#include <list>
#include <memory>

namespace DataStructures
{
    class ProcessingData;
}

namespace Config
{
    class JsonConfig;
}

/**
 * @namespace Algorithms
 *
 * @brief Namespace of libalgorithms library.
 */
namespace Algorithms
{

/**
 * @class IAlgorithm
 *
 * @brief This is a base class for different types of algorithms.
 */
class IAlgorithm
{
public:
    /**
     * @brief Default constructor.
     */
    explicit IAlgorithm(const std::shared_ptr<Config::JsonConfig>& config){};

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    virtual bool Process(std::shared_ptr<DataStructures::ProcessingData>& processingData) = 0;


    /**
     * @brief Default destructor.
     */
    virtual ~IAlgorithm() = default;
};

}
#endif // INTERFACE_ALGORITHM_H