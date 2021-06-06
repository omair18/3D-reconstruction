#include <filesystem>

#include "PointCloudDensificationAlgorithm.h"
#include "Logger.h"
#include "ConfigNodes.h"
#include "ModelDataset.h"
#include "ProcessingData.h"
#include "ReconstructionParams.h"
#include "Scene.h"
#include "JsonConfig.h"

namespace Algorithms
{

PointCloudDensificationAlgorithm::PointCloudDensificationAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
isInitialized_(false),
saveResult_(false),
resolutionLevel_(1),
maxResolution_(3200),
minResolution_(640),
minViews_(2),
maxViews_(12),
minViewsFuse_(3),
minViewsFilter_(2),
minViewsFilterAdjust_(1),
minViewsTrustPoint_(2),
numberOfViews_(5),
filterAdjust_(true),
addCorners_(true),
viewMinScore_(2.0f),
viewMinScoreRatio_(0.3f),
minArea_(0.05f),
minAngle_(3.0f),
optimalAngle_(10.0f),
maxAngle_(65.0f),
descriptorMinMagnitudeThreshold_(0.01f),
depthDiffThreshold_(0.01f),
normalDiffThreshold_(25.0f),
pairwiseMul_(0.3f),
optimizerEps_(0.001f),
optimizerMaxIterations_(80),
speckleSize_(100),
interpolationGapSize_(7),
optimize_(7),
estimateColors_(2),
estimateNormals_(2),
NCCThresholdKeep_(0.55f),
estimationIterations_(4),
randomIterations_(6),
randomMaxScale_(2),
randomDepthRatio_(0.003f),
randomAngle1Range_(16.0f),
randomAngle2Range_(10.0f),
randomSmoothDepth_(0.02f),
randomSmoothNormal_(13.0f),
randomSmoothBonus_(0.93f)
{
    InitializeInternal(config);
}

bool PointCloudDensificationAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    using namespace Log;

    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();
    auto& reconstructionParams = dataset->GetReconstructionParams();
    auto& scene = *reconstructionParams->scene_;

    for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
    << "Performing point cloud densification algorithm on dataset " << datasetUUID << " ...";

    if(scene.DenseReconstruction())
    {
        if (saveResult_)
        {
            scene.Save("scene_dense.mvs");
            scene.pointcloud.Save("scene_dense.ply");
        }
        return true;
    }
    else
    {
        return false;
    }
}

void PointCloudDensificationAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void PointCloudDensificationAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;
    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ResolutionLevel))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ResolutionLevel
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxResolution))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxResolution
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinResolution))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinResolution
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViews))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViews
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxViews))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxViews
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFuse))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFuse
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFilter))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFilter
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFilterAdjust))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFilterAdjust
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsTrustPoint))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsTrustPoint
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NumbersOfViews))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NumbersOfViews
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::FilterAdjust))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::FilterAdjust
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::AddCorners))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::AddCorners
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ViewMinScore))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ViewMinScore
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ViewMinScoreRatio))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ViewMinScoreRatio
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinArea))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinArea
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinAngle))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinAngle
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimalAngle))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimalAngle
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxAngle))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxAngle
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::DescriptorMinMagnitudeThreshold))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::DescriptorMinMagnitudeThreshold
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::DepthDiffThreshold))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::DepthDiffThreshold
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NormalDiffThreshold))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NormalDiffThreshold
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::PairwiseMul))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::PairwiseMul
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimizerEps))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimizerEps
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimizerMaxIterations))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimizerMaxIterations
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::SpeckleSize))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::SpeckleSize
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::InterpolationGapSize))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::InterpolationGapSize
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::Optimize))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::Optimize
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimateColors))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimateColors
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimateNormals))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimateNormals
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NCCThresholdKeep))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NCCThresholdKeep
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimationIterations))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimationIterations
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomIterations))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomIterations
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomMaxScale))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomMaxScale
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomDepthRatio))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomDepthRatio
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomAngle1Range))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomAngle1Range
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomAngle2Range))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomAngle2Range
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothDepth))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothDepth
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothNormal))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothNormal
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothBonus))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid point cloud densification algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothBonus
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::SaveResult))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid point cloud densification algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::SaveResult
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

}

void PointCloudDensificationAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;
    if(!isInitialized_)
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
        << "Initializing point cloud densification algorithm ...";

        ValidateConfig(config);

        resolutionLevel_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ResolutionLevel]->ToInt32();
        maxResolution_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxResolution]->ToInt32();
        minResolution_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinResolution]->ToInt32();

        minViews_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViews]->ToInt32();
        maxViews_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxViews]->ToInt32();
        minViewsFuse_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFuse]->ToInt32();
        minViewsFilter_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFilter]->ToInt32();
        minViewsFilterAdjust_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsFilterAdjust]->ToInt32();
        minViewsTrustPoint_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinViewsTrustPoint]->ToInt32();
        numberOfViews_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NumbersOfViews]->ToInt32();
        filterAdjust_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::FilterAdjust]->ToBool();
        addCorners_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::AddCorners]->ToBool();
        viewMinScore_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ViewMinScore]->ToFloat();
        viewMinScoreRatio_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::ViewMinScoreRatio]->ToFloat();

        minArea_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinArea]->ToFloat();
        minAngle_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MinAngle]->ToFloat();
        optimalAngle_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimalAngle]->ToFloat();
        maxAngle_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::MaxAngle]->ToFloat();
        descriptorMinMagnitudeThreshold_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::DescriptorMinMagnitudeThreshold]->ToFloat();
        depthDiffThreshold_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::DepthDiffThreshold]->ToFloat();
        normalDiffThreshold_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NormalDiffThreshold]->ToFloat();
        pairwiseMul_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::PairwiseMul]->ToFloat();

        optimizerEps_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimizerEps]->ToFloat();
        optimizerMaxIterations_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::OptimizerMaxIterations]->ToInt32();
        speckleSize_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::SpeckleSize]->ToInt32();
        interpolationGapSize_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::InterpolationGapSize]->ToInt32();
        optimize_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::Optimize]->ToInt32();
        estimateColors_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimateColors]->ToInt32();
        estimateNormals_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimateNormals]->ToInt32();

        NCCThresholdKeep_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::NCCThresholdKeep]->ToFloat();
        estimationIterations_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::EstimationIterations]->ToInt32();
        randomIterations_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomIterations]->ToInt32();
        randomMaxScale_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomMaxScale]->ToInt32();
        randomDepthRatio_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomDepthRatio]->ToFloat();
        randomAngle1Range_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomAngle1Range]->ToFloat();
        randomAngle2Range_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomAngle2Range]->ToFloat();
        randomSmoothDepth_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothDepth]->ToFloat();
        randomSmoothNormal_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothNormal]->ToFloat();
        randomSmoothBonus_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::RandomSmoothBonus]->ToFloat();

        saveResult_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::PointCloudDensificationAlgorithm::SaveResult]->ToBool();

        MVS::OPTDENSE::init();
        MVS::OPTDENSE::update();
        MVS::OPTDENSE::nResolutionLevel = resolutionLevel_;
        MVS::OPTDENSE::nMaxResolution = maxResolution_;
        MVS::OPTDENSE::nMinResolution = minResolution_;
        MVS::OPTDENSE::nMinViews = minViews_;
        MVS::OPTDENSE::nMaxViews = maxViews_;
        MVS::OPTDENSE::nMinViewsFuse = minViewsFuse_;
        MVS::OPTDENSE::nMinViewsFilter = minViewsFilter_;
        MVS::OPTDENSE::nMinViewsFilterAdjust = minViewsFilterAdjust_;
        MVS::OPTDENSE::nMinViewsTrustPoint = minViewsTrustPoint_;
        MVS::OPTDENSE::nNumViews = numberOfViews_;
        MVS::OPTDENSE::bFilterAdjust = filterAdjust_;
        MVS::OPTDENSE::bAddCorners = addCorners_;
        MVS::OPTDENSE::fViewMinScore = viewMinScore_;
        MVS::OPTDENSE::fViewMinScoreRatio = viewMinScoreRatio_;
        MVS::OPTDENSE::fMinArea = minArea_;
        MVS::OPTDENSE::fMinAngle = minAngle_;
        MVS::OPTDENSE::fOptimAngle = optimalAngle_;
        MVS::OPTDENSE::fMaxAngle = 65.0; maxAngle_;
        MVS::OPTDENSE::fDescriptorMinMagnitudeThreshold = descriptorMinMagnitudeThreshold_;
        MVS::OPTDENSE::fDepthDiffThreshold = depthDiffThreshold_;
        MVS::OPTDENSE::fNormalDiffThreshold = normalDiffThreshold_;
        MVS::OPTDENSE::fPairwiseMul = pairwiseMul_;
        MVS::OPTDENSE::fOptimizerEps = optimizerEps_;
        MVS::OPTDENSE::nOptimizerMaxIters = optimizerMaxIterations_;
        MVS::OPTDENSE::nSpeckleSize = speckleSize_;
        MVS::OPTDENSE::nIpolGapSize = interpolationGapSize_;
        MVS::OPTDENSE::nOptimize = optimize_;
        MVS::OPTDENSE::nEstimateColors = estimateColors_;
        MVS::OPTDENSE::nEstimateNormals = estimateNormals_;
        MVS::OPTDENSE::fNCCThresholdKeep = NCCThresholdKeep_;
        MVS::OPTDENSE::nEstimationIters = estimationIterations_;
        MVS::OPTDENSE::nRandomIters = randomIterations_;
        MVS::OPTDENSE::nRandomMaxScale = randomMaxScale_;
        MVS::OPTDENSE::fRandomDepthRatio = randomDepthRatio_;
        MVS::OPTDENSE::fRandomAngle1Range = randomAngle1Range_;
        MVS::OPTDENSE::fRandomAngle2Range = randomAngle2Range_;
        MVS::OPTDENSE::fRandomSmoothDepth = randomSmoothDepth_;
        MVS::OPTDENSE::fRandomSmoothNormal = randomSmoothNormal_;
        MVS::OPTDENSE::fRandomSmoothBonus = randomSmoothBonus_;

        Util::Init();

        isInitialized_ = true;
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
        << "Point cloud densification algorithm was successfully initialized";
    }
    else
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::WARNING, __PRETTY_FUNCTION__)
        << "Point cloud densification algorithm is already initialized.";
    }
}

}
