#include "WebServerFactory.h"
#include "WebServer.h"

namespace Networking
{

std::unique_ptr<WebServer> WebServerFactory::Create()
{
    return std::unique_ptr<WebServer>();
}

}