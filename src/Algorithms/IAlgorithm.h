/**
 * @file IAlgorithm.h.
 *
 * @brief Declares the IAlgorithm class. This is a base class for different types of algorithms.
 */

#ifndef INTERFACE_ALGORITHM_H
#define INTERFACE_ALGORITHM_H

#include <memory>

// forward declaration for DataStructures::ProcessingData
namespace DataStructures
{
    class ProcessingData;
}

// forward declaration for Config::JsonConfig
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
    explicit IAlgorithm() = default;

    /**
     * @brief
     *
     * @param processingData
     * @return
     */
    virtual bool Process(std::shared_ptr<DataStructures::ProcessingData>& processingData) = 0;

    /**
     * @brief
     *
     * @param config
     */
    virtual void Initialize(const std::shared_ptr<Config::JsonConfig>& config) = 0;

    /**
     * @brief Default destructor.
     */
    virtual ~IAlgorithm() = default;

    /**
     * @brief Checks weather this algorithm requires GPU.
     * @return True if GPU is required.
     */
    [[nodiscard]] bool RequiresGPU() const { return isGPURequired_; };

protected:

    ///
    bool isGPURequired_ = false;
};

}
#endif // INTERFACE_ALGORITHM_H