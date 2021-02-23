#include "DefaultJsonConfigGenerator.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"

namespace Config
{

std::shared_ptr<JsonConfig> DefaultJsonConfigGenerator::GenerateKafkaConsumerDefaultConfig()
{
    return std::shared_ptr<JsonConfig>();
}

std::shared_ptr<JsonConfig> DefaultJsonConfigGenerator::GenerateKafkaProducerDefaultConfig()
{
    return std::shared_ptr<JsonConfig>();
}

std::shared_ptr<JsonConfig> DefaultJsonConfigGenerator::GenerateServiceDefaultConfig()
{
    auto config = std::make_shared<JsonConfig>();
    auto webServerConfig = std::make_shared<JsonConfig>();
    auto gpuConfig = std::make_shared<JsonConfig>();
    auto pipelineConfig = std::make_shared<JsonConfig>();
    auto queueConfig = std::make_shared<JsonConfig>();

    webServerConfig->AddNodeBool(ConfigNodes::ServiceConfig::WebServerConfig::Enabled, true);
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::IpAddress, "127.0.0.1");
    webServerConfig->AddNodeInt(ConfigNodes::ServiceConfig::WebServerConfig::Port, 8751);
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::Protocol,
                                   ConfigNodes::ServiceConfig::WebServerConfig::HTTP);
    webServerConfig->AddNodeInt(ConfigNodes::ServiceConfig::WebServerConfig::ThreadPoolSize, 2);
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::CertificatePath, "/home/a.crt");
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::PublicKeyPath, "/home/b.key");
    webServerConfig->AddNodeBool(ConfigNodes::ServiceConfig::WebServerConfig::UseDhParams, true);
    webServerConfig->AddNodeString(ConfigNodes::ServiceConfig::WebServerConfig::DhParamsPath, "/home/c.pem");

    gpuConfig->AddNodeString(ConfigNodes::ServiceConfig::GpuConfig::SelectionPolicy,
                             ConfigNodes::ServiceConfig::GpuConfig::Manual);
    gpuConfig->AddNodeInt(ConfigNodes::ServiceConfig::GpuConfig::Id, 0);



    config->SetNode(ConfigNodes::ServiceConfig::WebServer, webServerConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Gpu, gpuConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Pipeline, pipelineConfig);
    config->SetNode(ConfigNodes::ServiceConfig::Queue, queueConfig);

    return config;
}


}