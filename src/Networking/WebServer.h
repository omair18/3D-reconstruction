/**
 * @file WebServer.h.
 *
 * @brief
 */

#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include <vector>
#include <thread>

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

/**
 * @namespace Networking
 *
 * @brief
 */
namespace Networking
{

// forward declaration for Networking::EndpointListener
class EndpointListener;

/**
 * @class WebServer
 *
 * @brief
 */
class WebServer
{

public:

    /**
     * @brief
     */
    WebServer() = default;

    /**
     * @brief
     */
    ~WebServer();

    /**
     * @brief
     *
     * @param serviceConfig
     */
    void Initialize(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

private:

    /**
     * @brief
     *
     * @param webServerConfig
     * @return
     */
    static bool ValidateServerConfiguration(const std::shared_ptr<Config::JsonConfig>& webServerConfig);

    ///
    std::string address_;

    ///
    int port_ = -1;

    ///
    std::vector<std::thread> threadPool_;

    ///
    std::string webDirectoryPath_;

    ///
    std::unique_ptr<EndpointListener> endpointListener_;
};

}

#endif // WEB_SERVER_H
