#ifndef WEB_SERVER_FACTORY_H
#define WEB_SERVER_FACTORY_H

#include <memory>

namespace Networking
{

class WebServer;

class WebServerFactory
{
public:
    static std::unique_ptr<WebServer> Create();
};

}

#endif // WEB_SERVER_FACTORY_H
