#include <filesystem>

#include "DefaultJsonConfigGenerator.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "PathUtils.h"

namespace Config
{

std::shared_ptr<JsonConfig> DefaultJsonConfigGenerator::GenerateServiceDefaultConfig()
{
    auto config = std::make_shared<JsonConfig>();
    auto webServerConfig = std::make_shared<JsonConfig>();
    auto gpuConfig = std::make_shared<JsonConfig>();
    auto pipelineConfig = std::make_shared<JsonConfig>();
    auto queueConfig = std::make_shared<JsonConfig>();
    std::filesystem::path defaultWebDirectoryPath = Utils::PathUtils::GetExecutableFolderPath();
    defaultWebDirectoryPath /= "web";

    webServerConfig->AddNodeBool(ConfigNodes::ServiceConfig::WebServerConfig::Enabled, true);
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::IpAddress, "127.0.0.1");
    webServerConfig->AddNodeInt(ConfigNodes::ServiceConfig::WebServerConfig::Port, 8751);
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::Protocol,
                                   ConfigNodes::ServiceConfig::WebServerConfig::HTTP);
    webServerConfig->AddNodeInt(ConfigNodes::ServiceConfig::WebServerConfig::ThreadPoolSize, 2);
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::WebFolderPath,
                                   defaultWebDirectoryPath.string());
    webServerConfig->AddNodeBool(ConfigNodes::ServiceConfig::WebServerConfig::UseDhParams, true);

    gpuConfig->AddNodeString(ConfigNodes::ServiceConfig::GpuConfig::SelectionPolicy,
                             ConfigNodes::ServiceConfig::GpuConfig::Manual);
    gpuConfig->AddNodeInt(ConfigNodes::ServiceConfig::GpuConfig::Id, 0);



    config->SetNode(ConfigNodes::ServiceConfig::WebServer, webServerConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Gpu, gpuConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Pipeline, pipelineConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Queues, queueConfig);

    return config;
}


}