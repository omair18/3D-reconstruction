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

    /**
     * @brief
     *
     * @return
     */
    static std::shared_ptr<JsonConfig> GenerateKafkaConsumerDefaultConfig();

    /**
     * @brief
     *
     * @return
     */
    static std::shared_ptr<JsonConfig> GenerateKafkaProducerDefaultConfig();

    /**
     * @brief
     *
     * @return
     */
    static std::shared_ptr<JsonConfig> GenerateServiceDefaultConfig();

};

}

#endif // DEFAULT_JSON_CONFIG_GENERATOR_H
