/**
 * @file ConfigNodes.h.
 *
 * @brief Declares constant configuration nodes in all configuration files.
 */


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

            const inline static std::string Pipeline = "pipeline";
            namespace PipelineConfig
            {
                const inline static std::string Name = "name";
                const inline static std::string Type = "type";
                const inline static std::string Cpu = "CPU";
                const inline static std::string Gpu = "GPU";
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

            const inline static std::string Storage = "storage";
            namespace StorageConfig
            {
                const inline static std::string& Type = PipelineConfig::Type;
                const inline static std::string Filesystem = "filesystem";
                const inline static std::string AmazonBucket = "amazonBucket";
                const inline static std::string Path = "path";
            }
        }

        namespace NetworkingConfig
        {
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

            namespace KafkaConsumerConfig
            {
                const inline static std::string Brokers = "brokers";
                const inline static std::string Topics = "topics";
                const inline static std::string EnablePartitionEOF = "enablePartitionEOF";
                const inline static std::string GroupId = "groupId";
                const inline static std::string Timeout = "timeout";
            }

            namespace KafkaProducerConfig
            {
                const inline static std::string& Topics = KafkaConsumerConfig::Topics;
                const inline static std::string& Brokers = KafkaConsumerConfig::Brokers;
                const inline static std::string& Timeout = KafkaConsumerConfig::Timeout;
            }
        }

        namespace AlgorithmsConfig
        {
            namespace AlgorithmsNames
            {
                // Kafka IO
                const inline static std::string KafkaConsumptionAlgorithm = "kafkaConsumptionAlgorithm";
                const inline static std::string KafkaProducingAlgorithm = "kafkaProducingAlgorithm";

                // Decoding
                const inline static std::string CpuImageDecodingAlgorithm = "CpuImageDecodingAlgorithm";
                const inline static std::string CUDAImageDecodingAlgorithm = "CUDAImageDecodingAlgorithm";

                // Image processing
                const inline static std::string CUDAConvolutionAlgorithm = "CUDAConvolutionAlgorithm";
                const inline static std::string CUDAResizeAlgorithm = "CUDAResizeAlgorithm";

                // Photogrammetry
                const inline static std::string DatasetCollectingAlgorithm = "datasetCollectingAlgorithm";
            }

            namespace KafkaConsumptionAlgorithmConfig
            {

            }

            namespace KafkaProducingAlgorithmConfig
            {

            }

            namespace CpuImageDecodingAlgorithmConfig
            {

            }

            namespace CUDAImageDecodingAlgorithmConfig
            {

            }

            namespace CUDAConvolutionAlgorithmConfig
            {

            }

            namespace CUDAResizeAlgorithmConfig
            {

            }


        }

        namespace MessageNodes
        {
            const inline static std::string CameraID = "cameraID";
            const inline static std::string Timestamp = "timestamp";
            const inline static std::string UUID = "UUID";
            const inline static std::string FrameID = "frameID";
            const inline static std::string framesTotal = "framesTotal";
            const inline static std::string FocalLength = "focalLength";
            const inline static std::string SensorSize = "sensorSize";
        }
    }
}

#endif