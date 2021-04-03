/**
 * @file WebServerManager.h.
 *
 * @brief
 */

#ifndef WEB_SERVER_MANAGER_H
#define WEB_SERVER_MANAGER_H

#include <memory>

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

// forward declaration for Networking::WebServer
class WebServer;

/**
 * @class WebServerManager
 *
 * @brief
 */
class WebServerManager
{
public:

    /**
     * @brief
     */
    WebServerManager();

    /**
     * @brief
     */
    ~WebServerManager();

    /**
     * @brief
     *
     * @param serviceConfig
     */
    void CreateWebServer(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

private:

    ///
    std::unique_ptr<WebServer> webServer_;

};

}

#endif // WEB_SERVER_MANAGER_H
