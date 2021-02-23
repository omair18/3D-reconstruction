#include <pwd.h>
#include <filesystem>

#include "ServiceSDK.h"
#include "StackTraceDumper.h"
#include "Logger.h"
#include "JsonConfigManager.h"
#include "DefaultJsonConfigGenerator.h"
#include "GpuManager.h"
#include "ConfigNodes.h"

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

    InitializeGpuManager();

    auto matchingGpu = gpuManager_->SelectMatchingGPU(serviceConfig);



}

ServiceSDK::~ServiceSDK()
{
    LOGGER_FREE();
}

void ServiceSDK::InitializeConfigFolderPath()
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

    std::filesystem::path homeFolderPath = homeFolder;

    homeFolderPath /= organizationName;
    homeFolderPath /= productName;
    homeFolderPath /= version;

    if(!std::filesystem::exists(homeFolderPath))
    {
        LOG_TRACE() << "Config directory with path " << homeFolderPath << " is not created. Trying to create it ...";
        std::error_code errorCode;
        std::filesystem::create_directories(homeFolderPath, errorCode);
        if(errorCode)
        {
            LOG_ERROR() << "Failed to create config directory with path " << homeFolderPath << ". Error code "
            << errorCode.value() << ": " << errorCode.message();
            throw std::runtime_error(errorCode.message());
        }
        else
        {
            LOG_TRACE() << "Successfully created config directory with path " << homeFolderPath << ".";
        }
    }
    else
    {
        LOG_TRACE() << "Using existing config directory: " << homeFolderPath << ".";
    }

    configPath_ = homeFolderPath.string();
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

    LOG_TRACE() << "Service configuration successfully";

    return serviceConfig;
}

void ServiceSDK::InitializeGpuManager()
{
    gpuManager_->UpdateCUDACapableDevicesList();
}


}