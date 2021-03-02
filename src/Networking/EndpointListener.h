#ifndef ENDPOINT_LISTENER_H
#define ENDPOINT_LISTENER_H

#include <boost/asio/io_context.hpp>
#include <boost/asio/ssl/context.hpp>

namespace Config
{
    class JsonConfig;
}

namespace Networking
{

class EndpointListener
{
public:
    explicit EndpointListener(const std::shared_ptr<Config::JsonConfig>& config);
    ~EndpointListener();
private:
    void InitializeIoContext(const std::shared_ptr<Config::JsonConfig>& config);

    boost::asio::io_context ioContext_;
    boost::asio::ssl::context sslIoContext_;

    std::string certificate_;
    std::string key_;
    std::string dhparam_;
};

}


#endif // ENDPOINT_LISTENER_H
