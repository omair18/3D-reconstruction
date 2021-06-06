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

        namespace ProjectSubdirectories
        {
            const inline static std::string Models = "models";
            const inline static std::string Web = "web";
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
                const inline static std::string AllowUnconfiguredChannels = "allowUnconfiguredChannels";
                const inline static std::string BinarizationCoefficients = "binarizationCoefficients";
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

            namespace AKAZEKeyPointDetectionAlgorithm
            {
                const inline static std::string Octaves = "octaves";
                const inline static std::string SublayersPerOctave = "sublayersPerOctave";
                const inline static std::string Threshold = "threshold";
                const inline static std::string AnisotropicDiffusionFunction = "anisotropicDiffusionFunction";
                const inline static std::string PeronaMalikG1DiffusionFunction = "PERONA_MALIK_G1_DIFFUSION_FUNCTION";
                const inline static std::string PeronaMalikG2DiffusionFunction = "PERONA_MALIK_G2_DIFFUSION_FUNCTION";
                const inline static std::string WeickertDiffusionFunction = "WEICKERT_DIFFUSION_FUNCTION";
                const inline static std::string CharbonnierDiffusionFunction = "CHARBONNIER_DIFFUSION_FUNCTION";
            }

            namespace CUDAAKAZEKeyPointDetectionAlgorithm
            {

            }

            namespace KeyPointMatchingAlgorithm
            {
                const inline static std::string DistanceRatio = "distanceRatio";
            }

            namespace CUDAKeyPointMatchingAlgorithm
            {

            }

            namespace CUDAKeyPointFilteringAlgorithm
            {

            }

            namespace KeyPointFilteringAlgorithm
            {
                const inline static std::string MaxIterations = "maxIterations";
                const inline static std::string EstimationPrecision = "estimationPrecision";
            }

            namespace CUDABundleAdjustmentAlgorithm
            {

            }

            namespace BundleAdjustmentAlgorithm
            {
                const inline static std::string UseMotionPrior = "useMotionPrior";
                const inline static std::string SaveResult = "saveResult";
                const inline static std::string UseConstantFocalLength = "useConstantFocalLength";
                const inline static std::string UseConstantPrincipalPoint = "useConstantPrincipalPoint";
                const inline static std::string UseConstantDistortionParams = "useConstantDistortionParams";
                const inline static std::string TriangulationMethod = "triangulationMethod";
                const inline static std::string DirectLinearTransformTriangulationMethod = "DIRECT_LINEAR_TRANSFORM";
                const inline static std::string L1AngularTriangulationMethod = "L1_ANGULAR";
                const inline static std::string LInfinityAngularTriangulationMethod = "LINFINITY_ANGULAR";
                const inline static std::string InverseDepthWeightedMidpointTriangulationMethod = "INVERSE_DEPTH_WEIGHTED_MIDPOINT";
                const inline static std::string ResectionMethod = "resectionMethod";
                const inline static std::string DirectLinearTransform6Points = "DIRECT_LINEAR_TRANSFORM_6_POINTS";
                const inline static std::string P3P_KE_CVPR17 = "P3P_KE_CVPR17";
                const inline static std::string P3P_KNEIP_CVPR11 = "P3P_KNEIP_CVPR11";
                const inline static std::string P3P_NORDBERG_ECCV18 = "P3P_NORDBERG_ECCV18";
                const inline static std::string UP2P_KUKELOVA_ACCV10 = "UP2P_KUKELOVA_ACCV10";
            }

            namespace CUDAPointCloudDensificationAlgorithm
            {

            }

            namespace PointCloudDensificationAlgorithm
            {
                const inline static std::string& SaveResult = BundleAdjustmentAlgorithm::SaveResult;
                const inline static std::string ResolutionLevel = "resolutionLevel";
                const inline static std::string MaxResolution = "maxResolution";
                const inline static std::string MinResolution = "minResolution";
                const inline static std::string MinViews = "minViews";
                const inline static std::string MaxViews = "maxViews";
                const inline static std::string MinViewsFuse = "minViewsFuse";
                const inline static std::string MinViewsFilter = "minViewsFilter";
                const inline static std::string MinViewsFilterAdjust = "minViewsFilterAdjust";
                const inline static std::string MinViewsTrustPoint = "minViewsTrustPoint";
                const inline static std::string NumbersOfViews = "numbersOfViews";
                const inline static std::string FilterAdjust = "filterAdjust";
                const inline static std::string AddCorners = "addCorners";
                const inline static std::string ViewMinScore = "viewMinScore";
                const inline static std::string ViewMinScoreRatio = "viewMinScoreRatio";
                const inline static std::string MinArea = "minArea";
                const inline static std::string MinAngle = "minAngle";
                const inline static std::string OptimalAngle = "optimalAngle";
                const inline static std::string MaxAngle = "maxAngle";
                const inline static std::string DescriptorMinMagnitudeThreshold = "descriptorMinMagnitudeThreshold";
                const inline static std::string DepthDiffThreshold = "depthDiffThreshold";
                const inline static std::string NormalDiffThreshold = "normalDiffThreshold";
                const inline static std::string PairwiseMul = "pairwiseMul";
                const inline static std::string OptimizerEps = "optimizerEps";
                const inline static std::string OptimizerMaxIterations = "optimizerMaxIterations";
                const inline static std::string SpeckleSize = "speckleSize";
                const inline static std::string InterpolationGapSize = "interpolationGapSize";
                const inline static std::string Optimize = "optimize";
                const inline static std::string EstimateColors = "estimateColors";
                const inline static std::string EstimateNormals = "estimateNormals";
                const inline static std::string NCCThresholdKeep = "NCCThresholdKeep";
                const inline static std::string EstimationIterations = "estimationIterations";
                const inline static std::string RandomIterations = "randomIterations";
                const inline static std::string RandomMaxScale = "randomMaxScale";
                const inline static std::string RandomDepthRatio = "randomDepthRatio";
                const inline static std::string RandomAngle1Range = "randomAngle1Range";
                const inline static std::string RandomAngle2Range = "randomAngle2Range";
                const inline static std::string RandomSmoothDepth = "randomSmoothDepth";
                const inline static std::string RandomSmoothNormal = "randomSmoothNormal";
                const inline static std::string RandomSmoothBonus = "randomSmoothBonus";
            }

            namespace CUDAMeshReconstructionAlgorithm
            {

            }

            namespace MeshReconstructionAlgorithm
            {
                const inline static std::string& SaveResult = PointCloudDensificationAlgorithm::SaveResult;
                const inline static std::string DistanceInsert = "distanceInsert";
                const inline static std::string UseConstantWeight = "useConstantWeight";
                const inline static std::string UseFreeSpaceSupport = "useFreeSpaceSupport";
                const inline static std::string ThicknessFactor = "thicknessFactor";
                const inline static std::string QualityFactor = "qualityFactor";
                const inline static std::string DecimateMesh = "decimateMesh";
                const inline static std::string RemoveSpurious = "removeSpurious";
                const inline static std::string RemoveSpikes = "removeSpikes";
                const inline static std::string CloseHoles = "closeHoles";
                const inline static std::string SmoothMesh = "smoothMesh";
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
                const inline static std::string& SaveResult = PointCloudDensificationAlgorithm::SaveResult;
                const inline static std::string& ResolutionLevel = PointCloudDensificationAlgorithm::ResolutionLevel;
                const inline static std::string& MinResolution = PointCloudDensificationAlgorithm::MinResolution;
                const inline static std::string OutlierThreshold = "outlierThreshold";
                const inline static std::string RatioDataSmoothness = "ratioDataSmoothness";
                const inline static std::string GlobalSeamLeveling = "globalSeamLeveling";
                const inline static std::string LocalSeamLeveling = "localSeamLeveling";
                const inline static std::string TextureSizeMultiple = "textureSizeMultiple";
                const inline static std::string RectPackingHeuristic = "rectPackingHeuristic";
                const inline static std::string ColorEmpty = "colorEmpty";
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