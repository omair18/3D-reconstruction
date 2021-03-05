#include <pwd.h>
#include <unistd.h>
#include <climits>
#include <filesystem>

#include "PathUtils.h"

namespace Utils
{


std::string PathUtils::GetAppDataPath()
{
    char* homeFolder = getenv("HOME");
    if(!homeFolder)
    {
        struct passwd* pwd = getpwuid(getuid());
        if(pwd)
        {
            homeFolder = pwd->pw_dir;
        }
    }

    return std::string(homeFolder);
}

std::string PathUtils::GetExecutablePath()
{
    char buffer[PATH_MAX + 1];
    ssize_t pathLength = readlink("/proc/self/exe", buffer, PATH_MAX);
    if(pathLength == static_cast<ssize_t>(-1))
    {
        return std::string();
    }
    //add zero terminator
    buffer[pathLength]=0;

    return std::string (buffer);
}

std::string PathUtils::GetExecutableFolderPath()
{
    std::filesystem::path executablePath = GetExecutablePath();
    return executablePath.parent_path().string();
}

}