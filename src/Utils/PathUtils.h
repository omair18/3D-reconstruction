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

class PathUtils
{
public:
    static std::string GetAppDataPath();

    static std::string GetExecutablePath();

    static std::string GetExecutableFolderPath();
};

}
#endif // PATH_UTILS_H
