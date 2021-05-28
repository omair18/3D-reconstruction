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

        namespace DecodingConfig
        {
            const inline static std::string& Name = ServiceConfig::PipelineConfig::Name;
            namespace DecodersNames
            {
                const inline static std::string OpenCV = "OPENCV";
                const inline static std::string NvJPEG = "NVJPEG";
                const inline static std::string NvJPEGHardware = "NVJPEG_HARDWARE";
                const inline static std::string NvJPEG2K = "NVJPEG2K";
            }
        }

        namespace AlgorithmsConfig
        {

            const inline static std::string Algorithms = "algorithms";
            const inline static std::string& Name = ServiceConfig::PipelineConfig::Name;
            const inline static std::string Configuration = "configuration";

            namespace AlgorithmsNames
            {
                // Kafka IO
                const inline static std::string KafkaConsumptionAlgorithm = "kafkaConsumptionAlgorithm";
                const inline static std::string KafkaProducingAlgorithm = "kafkaProducingAlgorithm";
                const inline static std::string KafkaMessageParsingAlgorithm = "kafkaMessageParsingAlgorithm";

                // Data transferring
                const inline static std::string DeviceToHostTransferringAlgorithm = "deviceToHostTransferringAlgorithm";
                const inline static std::string HostToDeviceTransferringAlgorithm = "hostToDeviceTransferringAlgorithm";

                // Decoding
                const inline static std::string CpuImageDecodingAlgorithm = "cpuImageDecodingAlgorithm";
                const inline static std::string CUDAImageDecodingAlgorithm = "CUDAImageDecodingAlgorithm";

                // Image processing
                const inline static std::string CUDAImageConvolutionAlgorithm = "CUDAImageConvolutionAlgorithm";
                const inline static std::string ImageConvolutionAlgorithm = "imageConvolutionAlgorithm";
                const inline static std::string CUDAImageResizeAlgorithm = "CUDAImageResizeAlgorithm";
                const inline static std::string ImageResizeAlgorithm = "imageResizeAlgorithm";
                const inline static std::string CUDAImageBinarizationAlgorithm = "CUDAImageBinarizationAlgorithm";
                const inline static std::string ImageBinarizationAlgorithm = "imageBinarizationAlgorithm";
                const inline static std::string CUDAImageSeparableConvolutionAlgorithm = "CUDAImageSeparableConvolutionAlgorithm";
                const inline static std::string ImageSeparableConvolutionAlgorithm = "imageSeparableConvolutionAlgorithm";


                // Photogrammetry
                const inline static std::string DatasetCollectingAlgorithm = "datasetCollectingAlgorithm";

                const inline static std::string CUDAAKAZEKeyPointDetectionAlgorithm = "CUDAAKAZEKeyPointDetectionAlgorithm";
                const inline static std::string AKAZEKeyPointDetectionAlgorithm = "AKAZEKeyPointDetectionAlgorithm";

                const inline static std::string CUDAKeyPointMatchingAlgorithm = "CUDAKeyPointMatchingAlgorithm";
                const inline static std::string KeyPointMatchingAlgorithm = "keyPointMatchingAlgorithm";

                const inline static std::string CUDAKeyPointFilteringAlgorithm = "CUDAKeyPointFilteringAlgorithm";
                const inline static std::string KeyPointFilteringAlgorithm = "keyPointFilteringAlgorithm";

                const inline static std::string CUDABundleAdjustmentAlgorithm = "CUDABundleAdjustmentAlgorithm";
                const inline static std::string BundleAdjustmentAlgorithm = "bundleAdjustmentAlgorithm";

                const inline static std::string CUDAPointCloudDensificationAlgorithm = "CUDAPointCloudDensificationAlgorithm";
                const inline static std::string PointCloudDensificationAlgorithm = "pointCloudDensificationAlgorithm";

                const inline static std::string CUDAMeshReconstructionAlgorithm = "CUDAMeshReconstructionAlgorithm";
                const inline static std::string MeshReconstructionAlgorithm = "meshReconstructionAlgorithm";

                const inline static std::string CUDAMeshRefinementAlgorithm = "CUDAMeshRefinementAlgorithm";
                const inline static std::string MeshRefinementAlgorithm = "meshRefinementAlgorithm";

                const inline static std::string CUDAMeshTexturingAlgorithm = "CUDAMeshTexturingAlgorithm";
                const inline static std::string MeshTexturingAlgorithm = "meshTexturingAlgorithm";
            }

            namespace KafkaConsumptionAlgorithmConfig
            {
                const inline static std::string& Brokers = NetworkingConfig::KafkaConsumerConfig::Brokers;
                const inline static std::string& Topics = NetworkingConfig::KafkaConsumerConfig::Topics;
                const inline static std::string& EnablePartitionEOF = NetworkingConfig::KafkaConsumerConfig::EnablePartitionEOF;
                const inline static std::string& GroupId = NetworkingConfig::KafkaConsumerConfig::GroupId;
                const inline static std::string& Timeout = NetworkingConfig::KafkaConsumerConfig::Timeout;
            }

            namespace KafkaProducingAlgorithmConfig
            {
                const inline static std::string& Topics = NetworkingConfig::KafkaProducerConfig::Topics;
                const inline static std::string& Brokers = NetworkingConfig::KafkaProducerConfig::Brokers;
                const inline static std::string& Timeout = NetworkingConfig::KafkaProducerConfig::Timeout;
            }

            namespace KafkaMessageParsingAlgorithm
            {

            }

            namespace DeviceToHostTransferringAlgorithm
            {

            }

            namespace HostToDeviceTransferringAlgorithm
            {

            }

            namespace CpuImageDecodingAlgorithmConfig
            {
                const inline static std::string Decoders = "decoders";
                const inline static std::string RemoveSourceData = "removeSourceData";
            }

            namespace CUDAImageDecodingAlgorithmConfig
            {
                const inline static std::string& Decoders = CpuImageDecodingAlgorithmConfig::Decoders;
                const inline static std::string& RemoveSourceData = CpuImageDecodingAlgorithmConfig::RemoveSourceData;
            }

            namespace CUDAImageConvolutionAlgorithmConfig
            {

            }

            namespace ImageConvolutionAlgorithmConfig
            {

            }

            namespace CUDAImageResizeAlgorithmConfig
            {

            }

            namespace ImageResizeAlgorithmConfig
            {

            }

            namespace CUDAImageBinarizationAlgorithm
            {

            }

            namespace ImageBinarizationAlgorithm
            {

            }

            namespace CUDAImageSeparableConvolutionAlgorithm
            {

            }

            namespace ImageSeparableConvolutionAlgorithm
            {

            }

            namespace DatasetCollectingAlgorithm
            {
                const inline static std::string ExpireTimeout = "expireTimeout";
            }

            namespace CUDAAKAZEKeyPointDetectionAlgorithm
            {

            }

            namespace AKAZEKeyPointDetectionAlgorithm
            {

            }

            namespace CUDAKeyPointMatchingAlgorithm
            {

            }

            namespace KeyPointMatchingAlgorithm
            {

            }

            namespace CUDAKeyPointFilteringAlgorithm
            {

            }

            namespace KeyPointFilteringAlgorithm
            {

            }

            namespace CUDABundleAdjustmentAlgorithm
            {

            }

            namespace BundleAdjustmentAlgorithm
            {

            }

            namespace CUDAPointCloudDensificationAlgorithm
            {

            }

            namespace PointCloudDensificationAlgorithm
            {

            }

            namespace CUDAMeshReconstructionAlgorithm
            {

            }

            namespace MeshReconstructionAlgorithm
            {

            }

            namespace CUDAMeshRefinementAlgorithm
            {

            }

            namespace MeshRefinementAlgorithm
            {

            }

            namespace CUDAMeshTexturingAlgorithm
            {

            }

            namespace MeshTexturingAlgorithm
            {

            }

        }

        namespace MessageNodes
        {
            const inline static std::string CameraID = "cameraID";
            const inline static std::string Timestamp = "timestamp";
            const inline static std::string UUID = "UUID";
            const inline static std::string FrameID = "frameID";
            const inline static std::string FramesTotal = "framesTotal";
            const inline static std::string FocalLength = "focalLength";
            const inline static std::string SensorSize = "sensorSize";
            const inline static std::string DistortionFunctionID = "distortionFunctionID";
        }
    }
}

#endif