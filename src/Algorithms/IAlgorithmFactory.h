/**
 * @file IAlgorithmFactory.h.
 *
 * @brief Declares the IAlgorithmFactory class. This is a base class for different types of algorithms factories.
 */

#ifndef INTERFACE_ALGORITHM_FACTORY
#define INTERFACE_ALGORITHM_FACTORY

#include <memory>

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

// forward declaration for GPU::GpuManager
namespace GPU
{
    class GpuManager;
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
 * @brief
 */
class IAlgorithm;

/**
 * @class IAlgorithmFactory
 *
 * @brief This is a base class for different types of algorithms factories.
 */
class IAlgorithmFactory
{
public:
    /**
      * @brief Default constructor.
      */
    IAlgorithmFactory() = default;

    /**
      * @brief Default destructor.
      */
    virtual ~IAlgorithmFactory() = default;

    /**
      * @brief Initializes algorithm-param with a pointer to a specific algorithm-object.
      *
      * @param config - Pointer to algorithm's configuration params
      *
      * return Unique pointer to created algorithm.
      */
    virtual std::unique_ptr<IAlgorithm> Create(const std::shared_ptr<Config::JsonConfig>& config,
                                               const std::unique_ptr<GPU::GpuManager>& gpuManager,
                                               void* cudaStream) = 0;

};

}

#endif // INTERFACE_ALGORITHM_FACTORY