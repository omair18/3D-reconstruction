#include "MeshReconstructionAlgorithm.h"
#include "Logger.h"
#include "ProcessingData.h"
#include "ModelDataset.h"
#include "ConfigNodes.h"
#include "JsonConfig.h"
#include "ReconstructionParams.h"
#include "Scene.h"

namespace Algorithms
{

MeshReconstructionAlgorithm::MeshReconstructionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
isInitialized_(false),
saveResult_(false),
distanceInsert_(2.5f),
useConstantWeight_(true),
useFreeSpaceSupport_(false),
thicknessFactor_(1.0f),
qualityFactor_(2.0f),
decimateMesh_(1.0f),
removeSpurious_(30.0f),
removeSpikes_(true),
closeHoles_(30),
smoothMesh_(2)
{
    InitializeInternal(config);
}

bool MeshReconstructionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();
    auto& reconstructionParams = dataset->GetReconstructionParams();
    auto& scene = *reconstructionParams->scene_;

    if (useConstantWeight_)
    {
        scene.pointcloud.pointWeights.Release();
    }

    if(scene.ReconstructMesh(distanceInsert_, useFreeSpaceSupport_, 4, thicknessFactor_, qualityFactor_))
    {
        scene.mesh.Clean(decimateMesh_, removeSpurious_, removeSpikes_, closeHoles_, smoothMesh_, false);
        scene.mesh.Clean(1.f, 0.f, removeSpikes_, closeHoles_, 0, false); // extra cleaning trying to close more holes
        scene.mesh.Clean(1.f, 0.f, false, 0, 0,
                         true); // extra cleaning to remove non-manifold problems created by closing holes

        if (saveResult_)
        {
            scene.Save("scene_dense_mesh.mvs");
            scene.mesh.Save("scene_dense_mesh.ply");
        }
        return true;
    }
    else
    {
        return false;
    }
}

void MeshReconstructionAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void MeshReconstructionAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::SaveResult))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid mesh reconstruction algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::SaveResult
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::CloseHoles))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::CloseHoles
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::DecimateMesh))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::DecimateMesh
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::DistanceInsert))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::DistanceInsert
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::QualityFactor))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::QualityFactor
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::RemoveSpikes))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::RemoveSpikes
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::RemoveSpurious))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::RemoveSpurious
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::SmoothMesh))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::SmoothMesh
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::ThicknessFactor))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::ThicknessFactor
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::UseConstantWeight))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::UseConstantWeight
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::UseFreeSpaceSupport))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid mesh reconstruction algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::UseFreeSpaceSupport
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void MeshReconstructionAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;

    if (!isInitialized_)
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
        << "Initializing mesh reconstruction algorithm ...";

        ValidateConfig(config);

        saveResult_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::SaveResult]->ToBool();

        distanceInsert_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::DistanceInsert]->ToFloat();
        useConstantWeight_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::UseConstantWeight]->ToBool();
        useFreeSpaceSupport_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::UseFreeSpaceSupport]->ToBool();
        thicknessFactor_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::ThicknessFactor]->ToFloat();
        qualityFactor_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::QualityFactor]->ToFloat();
        decimateMesh_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::DecimateMesh]->ToFloat();
        removeSpurious_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::RemoveSpurious]->ToFloat();
        removeSpikes_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::RemoveSpikes]->ToBool();
        closeHoles_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::CloseHoles]->ToInt32();
        smoothMesh_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::MeshReconstructionAlgorithm::SmoothMesh]->ToInt32();

        isInitialized_ = true;
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
        << "Mesh reconstruction algorithm was successfully initialized.";
    }
    else
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::WARNING, __PRETTY_FUNCTION__)
        << "Mesh reconstruction algorithm is already initialized.";
    }
}

}