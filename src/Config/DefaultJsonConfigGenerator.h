/**
 * @file DefaultJsonConfigGenerator.h.
 *
 * @brief Declares the DefaultJsonConfigGenerator class. A class for generating default JSON configs.
 */


#ifndef DEFAULT_JSON_CONFIG_GENERATOR_H
#define DEFAULT_JSON_CONFIG_GENERATOR_H

#include <memory>

/**
 * @namespace Config
 *
 * @brief Namespace of libconfig library.
 */
namespace Config
{

// forward declaration for Config::JsonConfig
class JsonConfig;

/**
 * @class DefaultJsonConfigGenerator
 *
 * @brief A class for generating default JSON configs.
 */
class DefaultJsonConfigGenerator
{
public:

    /**
     * @brief Generates a default JSON configuration for service.
     *
     * @return Pointer to the generated config.
     */
    static std::shared_ptr<JsonConfig> GenerateServiceDefaultConfig();

    /**
     * @brief Generates a JSON queue configuration.
     *
     * @param name - Name of the queue
     * @param size - Size of the queue
     * @return Pointer to the generated config.
     */
    static std::shared_ptr<JsonConfig> GenerateQueueConfig(const std::string& name, size_t size);


};

}

#endif // DEFAULT_JSON_CONFIG_GENERATOR_H
