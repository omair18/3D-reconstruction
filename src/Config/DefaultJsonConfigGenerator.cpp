#include <filesystem>

#include "DefaultJsonConfigGenerator.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "PathUtils.h"

namespace Config
{

std::shared_ptr<JsonConfig> DefaultJsonConfigGenerator::GenerateServiceDefaultConfig()
{
    /// Creating nodes for service configuration
    auto config = std::make_shared<JsonConfig>();
    auto webServerConfig = std::make_shared<JsonConfig>();
    auto gpuConfig = std::make_shared<JsonConfig>();
    auto pipelineConfig = std::make_shared<JsonConfig>();
    auto queuesConfig = std::make_shared<JsonConfig>();
    auto resultStorageConfig = std::make_shared<JsonConfig>();

    /// web server node
    {
        std::filesystem::path defaultWebDirectoryPath = Utils::PathUtils::GetExecutableFolderPath();
        defaultWebDirectoryPath /= "web";

        webServerConfig->AddNodeBool(ConfigNodes::NetworkingConfig::WebServerConfig::Enabled, true);
        webServerConfig->AddNodeString(ConfigNodes::NetworkingConfig::WebServerConfig::IpAddress, "127.0.0.1");
        webServerConfig->AddNodeInt(ConfigNodes::NetworkingConfig::WebServerConfig::Port, 8751);
        webServerConfig->AddNodeString(ConfigNodes::NetworkingConfig::WebServerConfig::Protocol,
                                       ConfigNodes::NetworkingConfig::WebServerConfig::HTTP);
        webServerConfig->AddNodeInt(ConfigNodes::NetworkingConfig::WebServerConfig::ThreadPoolSize, 2);
        webServerConfig->AddNodeString(ConfigNodes::NetworkingConfig::WebServerConfig::WebFolderPath,
                                       defaultWebDirectoryPath.string());
        webServerConfig->AddNodeBool(ConfigNodes::NetworkingConfig::WebServerConfig::UseDhParams, true);
    }

    /// GPU node
    {
        gpuConfig->AddNodeString(ConfigNodes::ServiceConfig::GpuConfig::SelectionPolicy,
                                 ConfigNodes::ServiceConfig::GpuConfig::Manual);
        gpuConfig->AddNodeInt(ConfigNodes::ServiceConfig::GpuConfig::Id, 0);
    }

    /// Pipeline node
    {
        pipelineConfig->FromJsonString("[]");
    }

    /// Queues node
    {
        queuesConfig->FromJsonString("[]");
        queuesConfig->AddObject(GenerateQueueConfig("incomingMessages", 1500));
        queuesConfig->AddObject(GenerateQueueConfig("collectedDatasets", 1500));

    }



    config->SetNode(ConfigNodes::ServiceConfig::WebServer, webServerConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Gpu, gpuConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Pipeline, pipelineConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Queues, queuesConfig);

    return config;
}

std::shared_ptr<JsonConfig> DefaultJsonConfigGenerator::GenerateQueueConfig(const std::string& name, size_t size)
{
    auto queueConfig = std::make_shared<Config::JsonConfig>();
    queueConfig->AddNodeString(ConfigNodes::ServiceConfig::QueueConfig::Name, name);
    queueConfig->AddNodeInt(ConfigNodes::ServiceConfig::QueueConfig::Size, size);
    return queueConfig;
}


}