/**
 * @file UUIDGenerator.h.
 *
 * @brief Declares the class for generating UUIDs.
 */

#ifndef UUID_GENERATOR_H
#define UUID_GENERATOR_H

#include <string>

/**
 * @namespace Utils
 *
 * @brief Namespace of libutils library.
 */
namespace Utils
{

/**
 * @class UUIDGenerator
 *
 * @brief A class for generating UUIDs.
 */
class UUIDGenerator
{
public:

    /**
     * @brief Generates UUID according to RFC-4122 as std::string.
     *
     * @return Generated UUID as std::string.
     */
    static std::string GenerateUUID();
};

}

#endif // UUID_GENERATOR_H
