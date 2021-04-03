/**
 * @file ServiceSDK.h.
 *
 * @brief
 */

#ifndef SERVICE_SDK_H
#define SERVICE_SDK_H

#include <exception>
#include <memory>

// forward declaration for Config::JsonConfig and Config::JsonConfigManager
namespace Config
{
    class JsonConfig;
    class JsonConfigManager;
}

// forward declaration for GPU::GpuManager
namespace GPU
{
    class GpuManager;
}

// forward declaration for Networking::WebServerManager
namespace Networking
{
    class WebServerManager;
}

// forward declaration for DataStructures::ProcessingQueueManager
namespace DataStructures
{
    class ProcessingQueueManager;
}

// forward declaration for Processing::ProcessorManager
namespace Processing
{
    class ProcessorManager;
}

/**
 * @namespace Service
 *
 * @brief
 */
namespace Service
{

/**
 * @class ServiceSDK
 *
 * @brief
 */
class ServiceSDK final
{

public:

    /**
     * @brief
     *
     * @param argc
     * @param argv
     */
    ServiceSDK(int argc, char** argv);

    /**
     * @brief
     */
    ~ServiceSDK();

    /**
     * @brief
     */
    void Initialize();

    /**
     * @brief
     */
    void Start();

private:

    /**
     * @brief
     */
    void InitializeConfigFolderPath();

    /**
     * @brief
     */
    void InitializeConfigManager();

    /**
     * @brief
     */
    void InitializeGpuManager();

    /**
     * @brief
     *
     * @param serviceConfig
     */
    void InitializeServiceGPU(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    /**
     * @brief
     *
     * @param serviceConfig
     */
    void InitializeWebServer(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    /**
     * @brief
     *
     * @param serviceConfig
     */
    void InitializeProcessingQueues(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    /**
     * @brief
     *
     * @param serviceConfig
     */
    void InitializeProcessors(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    /**
     * @brief
     *
     * @return
     */
    std::shared_ptr<Config::JsonConfig> GetServiceConfig();

    /**
     * @brief
     *
     * @param serviceConfig
     */
    static void ValidateServiceConfiguration(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    ///
    constexpr const inline static char* organizationName = "BSUIR";

    ///
    constexpr const inline static char* productName = "3D-reconstruction";

    ///
    constexpr const inline static char* version = "v0.0.1";

    ///
    std::unique_ptr<Config::JsonConfigManager> configManager_;

    ///
    std::unique_ptr<GPU::GpuManager> gpuManager_;

    ///
    std::unique_ptr<Networking::WebServerManager> webServerManager_;

    ///
    std::unique_ptr<DataStructures::ProcessingQueueManager> queueManager_;

    ///
    std::unique_ptr<Processing::ProcessorManager>  processorManager_;

    ///
    std::string configPath_;

};

}

#endif // SERVICE_SDK_H
