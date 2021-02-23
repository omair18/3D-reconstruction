#ifndef SERVICE_SDK_H
#define SERVICE_SDK_H

#include <exception>
#include <memory>

namespace Config
{
    class JsonConfig;
    class JsonConfigManager;
}

namespace GPU
{
    class GpuManager;
}

namespace Service
{

class ServiceSDK
{
public:

    ServiceSDK(int argc = 0, char** argv = nullptr);

    ~ServiceSDK();

    void Initialize();

private:

    void InitializeConfigFolderPath();

    void InitializeConfigManager();

    void InitializeGpuManager();

    std::shared_ptr<Config::JsonConfig> GetServiceConfig();

    constexpr const inline static char* organizationName = "BSUIR";
    constexpr const inline static char* productName = "3D-reconstruction";
    constexpr const inline static char* version = "v0.0.1";

    std::unique_ptr<Config::JsonConfigManager> configManager_;
    std::unique_ptr<GPU::GpuManager> gpuManager_;

    std::string configPath_;

};

}

#endif // SERVICE_SDK_H
