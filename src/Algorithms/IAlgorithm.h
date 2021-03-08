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
    IAlgorithm() = default;

    /**
     * @brief Executes an algorithm on the container of ProcessingData.
     *
     * @param processingDataBatch - List of ProcessingData
     *
     * @return True if after processing the container contains at least one element, false otherwise.
     */
    virtual bool Process(std::shared_ptr<DataStructures::ProcessingData>& processingDataBatch,
                         bool usesCUDAStream = false, void* cudaStream = nullptr) = 0;


    /**
     * @brief Default destructor.
     */
    virtual ~IAlgorithm() = default;
};

}
#endif // INTERFACE_ALGORITHM_H