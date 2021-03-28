#include "WebServerManager.h"
#include "WebServer.h"
#include "EndpointListener.h"

namespace Networking
{

void WebServerManager::CreateWebServer(const std::shared_ptr<Config::JsonConfig> &serviceConfig)
{
    webServer_ = std::make_unique<WebServer>();
    webServer_->Initialize(serviceConfig);
}

WebServerManager::WebServerManager()
{

}

WebServerManager::~WebServerManager()
{

}

}