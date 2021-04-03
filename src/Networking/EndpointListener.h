/**
 * @file EndpointListener.h.
 *
 * @brief
 */

#ifndef ENDPOINT_LISTENER_H
#define ENDPOINT_LISTENER_H

#include <boost/asio/io_context.hpp>
#include <boost/asio/ssl/context.hpp>

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

/**
 * @class EndpointListener
 *
 * @brief
 */
class EndpointListener
{

public:

    /**
     * @brief
     *
     * @param config
     */
    explicit EndpointListener(const std::shared_ptr<Config::JsonConfig>& config);

    /**
     * @brief
     */
    ~EndpointListener();

private:

    /**
     * @brief
     *
     * @param config
     */
    void InitializeIoContext(const std::shared_ptr<Config::JsonConfig>& config);

    ///
    boost::asio::io_context ioContext_;

    ///
    boost::asio::ssl::context sslIoContext_;

    ///
    std::string certificate_;

    ///
    std::string key_;

    ///
    std::string dhparam_;
};

}


#endif // ENDPOINT_LISTENER_H
