#include <filesystem>
#include <fstream>

#include "EndpointListener.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace Networking
{

EndpointListener::EndpointListener(const std::shared_ptr<Config::JsonConfig> &config) :
        ioContext_((*config)[Config::ConfigNodes::NetworkingConfig::WebServerConfig::ThreadPoolSize]->ToInt32()),
        sslIoContext_(boost::asio::ssl::context::method::tlsv12)
{
    InitializeIoContext(config);
}

void EndpointListener::InitializeIoContext(const std::shared_ptr<Config::JsonConfig> &config)
{
    std::filesystem::path webDirectoryPath = (*config)[Config::ConfigNodes::NetworkingConfig::WebServerConfig::WebFolderPath]->ToString();
    bool useSsl = (*config)[Config::ConfigNodes::NetworkingConfig::WebServerConfig::HTTPS] ||
            (*config)[Config::ConfigNodes::NetworkingConfig::WebServerConfig::HTTP_AND_HTTPS];
    if(useSsl)
    {
        const std::string certificateExtension = ".crt";
        const std::string privateKeyExtension = ".key";
        const std::string dhParamExtension = ".pem";

        std::filesystem::path certificatesFolderPath = webDirectoryPath / "certificates";
        std::filesystem::path certificatePath;
        std::filesystem::path privateKeyPath;
        std::filesystem::path dhParamPath;

        for (auto& file : std::filesystem::directory_iterator(certificatesFolderPath))
        {
            if(file.path().extension().string() == certificateExtension)
            {
                certificatePath = file;
            }

            if(file.path().extension().string() == privateKeyExtension)
            {
                privateKeyPath = file;
            }

            if(file.path().extension().string() == dhParamExtension)
            {
                dhParamPath = file;
            }
        }

        bool useDhParams = (*config)[Config::ConfigNodes::NetworkingConfig::WebServerConfig::UseDhParams]->ToBool();

        std::ifstream certificateFile(certificatePath);

        std::ifstream privateKeyFile(privateKeyPath);

        std::string certificate((std::istreambuf_iterator<char>(certificateFile)), std::istreambuf_iterator<char>());

        std::string privateKey((std::istreambuf_iterator<char>(privateKeyFile)), std::istreambuf_iterator<char>());

        certificateFile.close();
        privateKeyFile.close();

        std::string dhParams;

        if(useDhParams)
        {
            std::ifstream dhParamsFile(dhParamPath);
            dhParams = std::string((std::istreambuf_iterator<char>(dhParamsFile)), std::istreambuf_iterator<char>());
            dhParamsFile.close();
        }
        sslIoContext_.set_options(
                boost::asio::ssl::context::default_workarounds |
                boost::asio::ssl::context::no_sslv2 |
                boost::asio::ssl::context::single_dh_use);


        sslIoContext_.set_password_callback([](std::size_t,
                                               boost::asio::ssl::context_base::password_purpose)
                                           {
                                               return "print_elements_kernel";
                                           });
        sslIoContext_.use_certificate_chain(
                boost::asio::buffer(certificate.data(), certificate.size()));

        sslIoContext_.use_private_key(
                boost::asio::buffer(privateKey.data(), privateKey.size()),
                boost::asio::ssl::context::file_format::pem);

        sslIoContext_.use_tmp_dh(
                boost::asio::buffer(dhParams.data(), dhParams.size()));
    }

}

EndpointListener::~EndpointListener()
{

}

}