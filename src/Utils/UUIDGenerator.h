#ifndef UUID_GENERATOR_H
#define UUID_GENERATOR_H

#include <string>

/**
 * @namespace Utils
 *
 * @brief Namespace of libutils library
 */
namespace Utils
{

/**
 * @class UUIDGenerator
 *
 * @brief
 */
class UUIDGenerator
{
public:

    /**
     * @brief
     *
     * @return
     */
    static std::string GenerateUUID();
};

}

#endif // UUID_GENERATOR_H
