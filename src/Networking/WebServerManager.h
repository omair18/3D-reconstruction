#ifndef WEB_SERVER_MANAGER_H
#define WEB_SERVER_MANAGER_H

#include <memory>

namespace Config
{
    class JsonConfig;
}

namespace Networking
{

class WebServer;

class WebServerManager
{
public:
    WebServerManager();

    ~WebServerManager();
    void CreateWebServer(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

private:

    std::unique_ptr<WebServer> webServer_;

};

}

#endif // WEB_SERVER_MANAGER_H
