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

    if(!webServerConfig.Contains(Config::ConfigNodes::ServiceConfig::WebServerConfig::IpAddress))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::ServiceConfig::WebServerConfig::IpAddress << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::ServiceConfig::WebServerConfig::Port))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::ServiceConfig::WebServerConfig::Port << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::ServiceConfig::WebServerConfig::Protocol))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::ServiceConfig::WebServerConfig::Protocol << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::ServiceConfig::WebServerConfig::ThreadPoolSize))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::ServiceConfig::WebServerConfig::ThreadPoolSize << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::ServiceConfig::WebServerConfig::WebFolderPath))
    {
        LOG_ERROR();
        throw std::runtime_error("Invalid web server configuration.");
    }

    if(!webServerConfig.Contains(Config::ConfigNodes::ServiceConfig::WebServerConfig::UseDhParams))
    {
        LOG_ERROR() << "Invalid web server configuration. There is no node "
        << Config::ConfigNodes::ServiceConfig::WebServerConfig::WebFolderPath << " in web server configuration.";
        throw std::runtime_error("Invalid web server configuration.");
    }

    return false;
}

void WebServer::Initialize(const std::shared_ptr<Config::JsonConfig> &webServerConfig)
{
    LOG_TRACE() << "Initializing web server ...";
    if(ValidateServerConfiguration(webServerConfig))
    {
        address_ = (*webServerConfig)[Config::ConfigNodes::ServiceConfig::WebServerConfig::IpAddress]->ToString();
        port_ = (*webServerConfig)[Config::ConfigNodes::ServiceConfig::WebServerConfig::Port]->ToInt();

    }
}

}
