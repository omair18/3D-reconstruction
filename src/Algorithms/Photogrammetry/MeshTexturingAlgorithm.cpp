#include "MeshTexturingAlgorithm.h"
#include "Logger.h"
#include "ProcessingData.h"
#include "ModelDataset.h"
#include "ConfigNodes.h"
#include "JsonConfig.h"
#include "ReconstructionParams.h"
#include "Scene.h"

namespace Algorithms
{

MeshTexturingAlgorithm::MeshTexturingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
isInitialized_(false),
saveResult_(false),
resolutionLevel_(0),
minResolution_(640),
outlierThreshold_(6e-2f),
ratioDataSmoothness_(0.1f),
globalSeamLeveling_(true),
localSeamLeveling_(true),
textureSizeMultiple_(0),
rectPackingHeuristic_(0),
colorEmpty_(16744231)
{
    InitializeInternal(config);
}

bool MeshTexturingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();
    auto& reconstructionParams = dataset->GetReconstructionParams();
    auto& scene = *reconstructionParams->scene_;

    if(scene.TextureMesh(resolutionLevel_, minResolution_, outlierThreshold_, ratioDataSmoothness_, globalSeamLeveling_,
                      localSeamLeveling_, textureSizeMultiple_, rectPackingHeuristic_, Pixel8U(static_cast<uint32_t>(colorEmpty_))))
    {
        if (saveResult_)
        {
            scene.Save("scene_dense_mesh_textured.mvs");
            scene.mesh.Save("scene_dense_mesh_textured.ply");
        }
        return true;
    }
    else
    {
        return false;
    }
}

void MeshTexturingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void MeshTexturingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;
    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::SaveResult))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::SaveResult
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::ColorEmpty))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::ColorEmpty
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::GlobalSeamLeveling))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::GlobalSeamLeveling
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::LocalSeamLeveling))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::LocalSeamLeveling
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::OutlierThreshold))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::OutlierThreshold
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::RatioDataSmoothness))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::RatioDataSmoothness
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::RectPackingHeuristic))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::RectPackingHeuristic
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::TextureSizeMultiple))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::TextureSizeMultiple
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::MinResolution))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::MinResolution
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::ResolutionLevel))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh texturing algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::ResolutionLevel
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void MeshTexturingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;

    if(!isInitialized_)
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
        << "Initializing mesh texturing algorithm ...";
        ValidateConfig(config);

        saveResult_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::SaveResult]->ToBool();
        resolutionLevel_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::ResolutionLevel]->ToInt32();
        minResolution_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::MinResolution]->ToInt32();
        outlierThreshold_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::OutlierThreshold]->ToFloat();
        ratioDataSmoothness_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::RatioDataSmoothness]->ToFloat();
        globalSeamLeveling_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::GlobalSeamLeveling]->ToBool();
        localSeamLeveling_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::LocalSeamLeveling]->ToBool();
        textureSizeMultiple_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::TextureSizeMultiple]->ToInt32();
        rectPackingHeuristic_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::RectPackingHeuristic]->ToInt32();
        colorEmpty_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshTexturingAlgorithm::ColorEmpty]->ToInt32();

        isInitialized_ = true;
    }
    else
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::WARNING, __PRETTY_FUNCTION__)
        << "Mesh texturing algorithm is already initialized.";
    }
}

}