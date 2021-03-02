#include <pwd.h>
#include <unistd.h>
#include <climits>
#include <filesystem>

#include "PathUtils.h"

namespace Utils
{


std::string PathUtils::GetAppDataPath()
{
#ifdef _WIN32
    wchar_t path[MAX_PATH];
    std::wstring app_data_path;
    if(SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_LOCAL_APPDATA | CSIDL_FLAG_CREATE, NULL, 0, path)))
    {
        app_data_path = path;
    }
    else
    {
        int err = GetLastError();
        LOG_ERROR("Failed to retrieve APPDATA path, error: %d", err);
    }

    return app_data_path;
#else
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
#endif
}

std::string PathUtils::GetExecutablePath()
{
#ifdef _WIN32

    TCHAR sPath[500];

    memset(sPath, 0, 500*sizeof(TCHAR));

    int nRet=::GetModulefile_name(NULL, sPath, 500);

    assert(nRet);

    std::wstring strPath(sPath);
    //some system processes have such leading characters in image path
    //(e.g. Winlogon.exe where we have our resident)
    if(strPath.find(L"\\??\\")==0)
        strPath=strPath.substr(4);

    return strPath;

#else
    char buffer[PATH_MAX + 1];
    ssize_t pathLength = readlink("/proc/self/exe", buffer, PATH_MAX);
    if(pathLength == static_cast<ssize_t>(-1))
    {
        return std::string();
    }
    //add zero terminator
    buffer[pathLength]=0;

    return std::string (buffer);
#endif

}

std::string PathUtils::GetExecutableFolderPath()
{
    std::filesystem::path executablePath = GetExecutablePath();
    return executablePath.parent_path().string();
}

}