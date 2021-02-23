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
 * @brief
 */
class DefaultJsonConfigGenerator
{
public:
    static std::shared_ptr<JsonConfig> GenerateKafkaConsumerDefaultConfig();

    static std::shared_ptr<JsonConfig> GenerateKafkaProducerDefaultConfig();

    static std::shared_ptr<JsonConfig> GenerateServiceDefaultConfig();

};

}

#endif // DEFAULT_JSON_CONFIG_GENERATOR_H
