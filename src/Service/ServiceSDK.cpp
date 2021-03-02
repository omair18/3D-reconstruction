#include <filesystem>

#include "ServiceSDK.h"
#include "StackTraceDumper.h"
#include "Logger.h"
#include "JsonConfigManager.h"
#include "JsonConfig.h"
#include "DefaultJsonConfigGenerator.h"
#include "GpuManager.h"
#include "ConfigNodes.h"
#include "PathUtils.h"
#include "WebServer.h"
#include "WebServerFactory.h"

namespace Service
{

ServiceSDK::ServiceSDK(int argc, char** argv) :
configManager_(std::make_unique<Config::JsonConfigManager>()),
gpuManager_(std::make_unique<GPU::GpuManager>())
{

}

void ServiceSDK::Initialize()
{
    LOGGER_INIT();

    Utils::StackTraceDumper::ProcessSignal(SIGABRT);
    Utils::StackTraceDumper::ProcessSignal(SIGSEGV);

    LOG_TRACE() << "Initializing service ...";

    InitializeConfigFolderPath();

    InitializeConfigManager();

    auto serviceConfig = GetServiceConfig();

    InitializeWebServer(serviceConfig);

    InitializeGpuManager();

    InitializeServiceGPU(serviceConfig);

}

ServiceSDK::~ServiceSDK()
{
    LOGGER_FREE();
}

void ServiceSDK::InitializeConfigFolderPath()
{
    std::filesystem::path appDataFolderPath = Utils::PathUtils::GetAppDataPath();

    appDataFolderPath /= organizationName;
    appDataFolderPath /= productName;
    appDataFolderPath /= version;

    if(!std::filesystem::exists(appDataFolderPath))
    {
        LOG_TRACE() << "Config directory with path " << appDataFolderPath << " is not created. Trying to create it ...";
        std::error_code errorCode;
        std::filesystem::create_directories(appDataFolderPath, errorCode);
        if(errorCode)
        {
            LOG_ERROR() << "Failed to create config directory with path " << appDataFolderPath << ". Error code "
                        << errorCode.value() << ": " << errorCode.message();
            throw std::runtime_error(errorCode.message());
        }
        else
        {
            LOG_TRACE() << "Successfully created config directory with path " << appDataFolderPath << ".";
        }
    }
    else
    {
        LOG_TRACE() << "Using existing config directory: " << appDataFolderPath << ".";
    }

    configPath_ = appDataFolderPath.string();
}

void ServiceSDK::InitializeConfigManager()
{
    LOG_TRACE() << "Reading configs from " << configPath_;
    configManager_->ReadSettings(configPath_);
}

std::shared_ptr<Config::JsonConfig> ServiceSDK::GetServiceConfig()
{
    if(!configManager_->ConfigExists(Config::ConfigNodes::ConfigFileNames::ServiceConfigurationFilename))
    {
        LOG_WARNING() << "Service configuration file not found in " << configPath_ << ". Creating a default one.";
        configManager_->SetConfig(Config::ConfigNodes::ConfigFileNames::ServiceConfigurationFilename,
                                  Config::DefaultJsonConfigGenerator::GenerateServiceDefaultConfig());
        configManager_->Save(Config::ConfigNodes::ConfigFileNames::ServiceConfigurationFilename);
    }

    auto serviceConfig = configManager_->GetConfig(Config::ConfigNodes::ConfigFileNames::ServiceConfigurationFilename);

    if (!serviceConfig)
    {
        LOG_ERROR() << "Service configuration is invalid";
        throw std::runtime_error("Service configuration is invalid");
    }

    LOG_TRACE() << "Service configuration successfully loaded";

    return serviceConfig;
}

void ServiceSDK::InitializeGpuManager()
{
    gpuManager_->UpdateCUDACapableDevicesList();
}

void ServiceSDK::InitializeServiceGPU(const std::shared_ptr<Config::JsonConfig>& serviceConfig)
{
    auto matchingGpu = gpuManager_->SelectMatchingGPU(serviceConfig);
    gpuManager_->SetDevice(matchingGpu);
}

void ServiceSDK::InitializeWebServer(const std::shared_ptr<Config::JsonConfig> &serviceConfig)
{
    if(!serviceConfig->Contains(Config::ConfigNodes::ServiceConfig::WebServer))
    {
        LOG_ERROR() << "Invalid service configuration. There is no node "
        << Config::ConfigNodes::ServiceConfig::WebServer
        << " in service configuration file.";
        throw std::runtime_error("Invalid service configuration.");
    }

    auto webServerConfig = (*serviceConfig)[Config::ConfigNodes::ServiceConfig::WebServer];

    if(!webServerConfig->Contains(Config::ConfigNodes::ServiceConfig::WebServerConfig::Enabled))
    {
        LOG_ERROR() << "Invalid service web server configuration. There is no node "
        << Config::ConfigNodes::ServiceConfig::WebServerConfig::Enabled << " in web server configuration node.";
        throw std::runtime_error("Invalid service web server configuration.");
    }

    bool webServerEnabled = (*webServerConfig)[Config::ConfigNodes::ServiceConfig::WebServerConfig::Enabled]->ToBool();

    if(webServerEnabled)
    {
        LOG_TRACE() << "Web server is enabled.";
        webServer_ = Networking::WebServerFactory::Create();
        webServer_->Initialize(webServerConfig);
    }
    else
    {
        LOG_TRACE() << "Web server is disabled.";
    }
}


}