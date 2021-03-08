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
     * @brief
     *
     * @return
     */
    static std::shared_ptr<JsonConfig> GenerateServiceDefaultConfig();

};

}

#endif // DEFAULT_JSON_CONFIG_GENERATOR_H
