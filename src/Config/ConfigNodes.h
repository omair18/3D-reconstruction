/**--------------------------------------------------------------------------------------------------
 * @file	ConfigNodes.h.
 *
 * Declares constant configuration nodes in all configuration files.
 *-----------------------------------------------------------------------------------------------**/


#ifndef CONFIG_NODES_H
#define CONFIG_NODES_H

#include <string>

/**
 * @namespace Config
 *
 * @brief Namespace of libconfig library.
 */
namespace Config
{
    /**
     * @namespace ConfigNodes
     *
     * @brief Namespace of constant configuration nodes.
     */
    namespace ConfigNodes
    {
        namespace ConfigFileNames
        {
            namespace Default
            {
                const inline static std::string KafkaConsumer = "KafkaConsumer";
                const inline static std::string KafkaProducer = "KafkaProducer";
            }
            const inline static std::string ServiceConfigurationFilename = "service_configuration";
        }

        namespace ServiceConfig
        {
            const inline static std::string WebServer = "webServer";
            namespace WebServerConfig
            {
                const inline static std::string Enabled = "enabled";
                const inline static std::string IpAddress = "ipAddress";
                const inline static std::string Port = "port";
                const inline static std::string Protocol = "protocol";
                const inline static std::string ThreadPoolSize = "threadPoolSize";
                const inline static std::string HTTP = "HTTP";
                const inline static std::string HTTPS = "HTTPS";
                const inline static std::string HTTP_AND_HTTPS = "HTTP_AND_HTTPS";
                const inline static std::string UseDhParams = "useDhParams";
                const inline static std::string WebFolderPath = "webFolderPath";
            }

            const inline static std::string Pipeline = "pipeline";
            namespace PipelineConfig
            {
                const inline static std::string Name = "name";
                const inline static std::string InstancesCount = "instancesCount";
                const inline static std::string Input = "input";
                const inline static std::string Output = "output";
            }

            const inline static std::string Queues = "queues";
            namespace QueueConfig
            {
                constexpr const static std::string& Name = PipelineConfig::Name;
                const inline static std::string Size = "size";
            }

            const inline static std::string Gpu = "gpu";
            namespace GpuConfig
            {
                const inline static std::string SelectionPolicy = "selectionPolicy";
                const inline static std::string Manual = "MANUAL";
                const inline static std::string Id = "id";
                const inline static std::string Newest = "NEWEST";
                const inline static std::string MostCapacious = "MOST_CAPACIOUS";
                const inline static std::string Fastest = "FASTEST";
                const inline static std::string MaxSMs = "MAX_SMs";
            }

        }

    }
}

#endif