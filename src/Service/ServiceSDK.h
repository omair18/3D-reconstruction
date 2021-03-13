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

namespace Networking
{
    class WebServerManager;
}

namespace DataStructures
{
    class ProcessingQueueManager;
}

namespace Service
{

class ServiceSDK
{
public:

    ServiceSDK(int argc, char** argv);

    ~ServiceSDK();

    void Initialize();

    void Start();

private:

    void InitializeConfigFolderPath();

    void InitializeConfigManager();

    void InitializeGpuManager();

    void InitializeServiceGPU(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    void InitializeWebServer(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    void InitializeProcessingQueues(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    void InitializeProcessors(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    std::shared_ptr<Config::JsonConfig> GetServiceConfig();

    constexpr const inline static char* organizationName = "BSUIR";
    constexpr const inline static char* productName = "3D-reconstruction";
    constexpr const inline static char* version = "v0.0.1";

    std::unique_ptr<Config::JsonConfigManager> configManager_;
    std::unique_ptr<GPU::GpuManager> gpuManager_;
    std::unique_ptr<Networking::WebServerManager> webServerManager_;
    std::unique_ptr<DataStructures::ProcessingQueueManager> queueManager_;


    std::string configPath_;

};

}

#endif // SERVICE_SDK_H
