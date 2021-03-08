/**
 * @file PathUtils.h.
 *
 * @brief Declares the class for providing some useful paths.
 */

#ifndef PATH_UTILS_H
#define PATH_UTILS_H

#include <string>

/**
 * @namespace Utils
 *
 * @brief Namespace of libutils library.
 */
namespace Utils
{

/**
 * @class PathUtils
 *
 * @brief A the class for providing some useful paths.
 */
class PathUtils
{
public:

    /**
     * @brief Provides a path to home directory of current user.
     *
     * @return Home directory path of current user as std::string.
     */
    static std::string GetAppDataPath();

    /**
     * @brief Provides a path to an executable binary file.
     *
     * @return Path to executable binary file as std::string.
     */
    static std::string GetExecutablePath();

    /**
     * @brief Provides a path to a folder with executable file.
     *
     * @return A path to a folder with executable file as std::string.
     */
    static std::string GetExecutableFolderPath();
};

}
#endif // PATH_UTILS_H
