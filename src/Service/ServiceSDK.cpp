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
#include "WebServerManager.h"
#include "ProcessingQueueManager.h"
#include "ProcessorManager.h"

namespace Service
{

ServiceSDK::ServiceSDK(int argc, char** argv) :
configManager_(std::make_unique<Config::JsonConfigManager>()),
gpuManager_(std::make_unique<GPU::GpuManager>()),
webServerManager_(std::make_unique<Networking::WebServerManager>()),
queueManager_(std::make_unique<DataStructures::ProcessingQueueManager>()),
processorManager_(std::make_unique<Processing::ProcessorManager>())
{

}

void ServiceSDK::Initialize()
{
    LOGGER_INIT();

    Utils::StackTraceDumper::ProcessSignal(SIGABRT);
    Utils::StackTraceDumper::ProcessSignal(SIGSEGV);
    Utils::StackTraceDumper::ProcessSignal(SIGTERM);

    LOG_TRACE() << "Initializing service ...";

    InitializeConfigFolderPath();

    InitializeConfigManager();

    auto serviceConfig = GetServiceConfig();

    ValidateServiceConfiguration(serviceConfig);

    InitializeWebServer(serviceConfig);

    InitializeGpuManager();

    InitializeServiceGPU(serviceConfig);

    InitializeProcessingQueues(serviceConfig);

    InitializeProcessors(serviceConfig);

}

ServiceSDK::~ServiceSDK()
{
    processorManager_->StopAllProcessors();
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
    webServerManager_->CreateWebServer(serviceConfig);
}

void ServiceSDK::Start()
{
    LOG_TRACE() << "Starting service ...";
    processorManager_->StartAllProcessors();
}

void ServiceSDK::InitializeProcessingQueues(const std::shared_ptr<Config::JsonConfig> &serviceConfig)
{
    LOG_TRACE() << "Creating processing queues ...";
    queueManager_->Initialize(serviceConfig);
}

void ServiceSDK::InitializeProcessors(const std::shared_ptr<Config::JsonConfig>& serviceConfig)
{
    LOG_TRACE() << "Creating processors ...";
    processorManager_->Initialize(serviceConfig, configManager_, gpuManager_, queueManager_);
}

void ServiceSDK::ValidateServiceConfiguration(const std::shared_ptr<Config::JsonConfig> &serviceConfig)
{
    if(!serviceConfig->Contains(Config::ConfigNodes::ServiceConfig::WebServer))
    {
        LOG_ERROR() << "Invalid service configuration. There is no node "
                    << Config::ConfigNodes::ServiceConfig::WebServer << " in service configuration.";
        throw std::runtime_error("Invalid service configuration");
    }

    if(!serviceConfig->Contains(Config::ConfigNodes::ServiceConfig::Gpu))
    {
        LOG_ERROR() << "Invalid service configuration. There is no node "
                    << Config::ConfigNodes::ServiceConfig::Gpu << " in service configuration.";
        throw std::runtime_error("Invalid service configuration");
    }

    if(!serviceConfig->Contains(Config::ConfigNodes::ServiceConfig::Pipeline))
    {
        LOG_ERROR() << "Invalid service configuration. There is no node "
                    << Config::ConfigNodes::ServiceConfig::Pipeline << " in service configuration.";
        throw std::runtime_error("Invalid service configuration");
    }

    if(!serviceConfig->Contains(Config::ConfigNodes::ServiceConfig::Queues))
    {
        LOG_ERROR() << "Invalid service configuration. There is no node "
                    << Config::ConfigNodes::ServiceConfig::Queues << " in service configuration.";
        throw std::runtime_error("Invalid service configuration");
    }

    if(!serviceConfig->Contains(Config::ConfigNodes::ServiceConfig::Storage))
    {
        LOG_ERROR() << "Invalid service configuration. There is no node "
                    << Config::ConfigNodes::ServiceConfig::Storage << " in service configuration.";
        throw std::runtime_error("Invalid service configuration");
    }

}


}