#include "WebServer.h"
#include "EndpointListener.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace Networking
{

WebServer::~WebServer()
{

}

bool WebServer::ValidateServerConfiguration(const std::shared_ptr<Config::JsonConfig> &webServerConfig_)
{
    auto& webServerConfig = *webServerConfig_;

    if(!webServerConfig.Contains(Config::ConfigNodes::NetworkingConfig::WebServerConfig::IpAddress))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::NetworkingConfig::WebServerConfig::IpAddress << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::NetworkingConfig::WebServerConfig::Port))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::NetworkingConfig::WebServerConfig::Port << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::NetworkingConfig::WebServerConfig::Protocol))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::NetworkingConfig::WebServerConfig::Protocol << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::NetworkingConfig::WebServerConfig::ThreadPoolSize))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::NetworkingConfig::WebServerConfig::ThreadPoolSize << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::NetworkingConfig::WebServerConfig::WebFolderPath))
    {
        LOG_ERROR();
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::NetworkingConfig::WebServerConfig::UseDhParams))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::NetworkingConfig::WebServerConfig::WebFolderPath << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    return false;
}

void WebServer::Initialize(const std::shared_ptr<Config::JsonConfig> &serviceConfig)
{
    LOG_TRACE() << "Initializing web server ...";
    auto webServerConfig = (*serviceConfig)[Config::ConfigNodes::ServiceConfig::WebServer];
    if(ValidateServerConfiguration(webServerConfig))
    {
        address_ = (*webServerConfig)[Config::ConfigNodes::NetworkingConfig::WebServerConfig::IpAddress]->ToString();
        port_ = (*webServerConfig)[Config::ConfigNodes::NetworkingConfig::WebServerConfig::Port]->ToInt();

    }
}

}
