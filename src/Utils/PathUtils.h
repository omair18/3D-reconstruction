#ifndef PATH_UTILS_H
#define PATH_UTILS_H

#include <string>

/**
 * @namespace Utils
 *
 * @brief Namespace of libutils library
 */
namespace Utils
{

/**
 * @class PathUtils
 *
 * @brief
 */
class PathUtils
{
public:

    /**
     * @brief
     *
     * @return
     */
    static std::string GetAppDataPath();

    /**
     * @brief
     *
     * @return
     */
    static std::string GetExecutablePath();

    /**
     * @brief
     *
     * @return
     */
    static std::string GetExecutableFolderPath();
};

}
#endif // PATH_UTILS_H
