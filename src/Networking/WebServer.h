#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include <vector>
#include <thread>

namespace Config
{
    class JsonConfig;
}

namespace Networking
{


class EndpointListener;

class WebServer
{

public:
    WebServer() = default;

    void Initialize(const std::shared_ptr<Config::JsonConfig>& webServerConfig);

    ~WebServer();
private:

    static bool ValidateServerConfiguration(const std::shared_ptr<Config::JsonConfig>& webServerConfig);

    std::string address_;
    int port_ = -1;
    std::vector<std::thread> threadPool_;

    std::string webDirectoryPath_;

    std::unique_ptr<EndpointListener> endpointListener_;
};

}

#endif // WEB_SERVER_H
