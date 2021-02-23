#ifndef HTTPS_SERVER_H
#define HTTPS_SERVER_H

#include <vector>
#include <thread>

class HttpServer
{
public:

private:
    int port_;
    std::vector<std::thread> threadPool_;
    std::string certificate_;
    std::string key_;
    std::string dhparam_;
};


#endif // HTTPS_SERVER_H
