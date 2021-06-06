#include <filesystem>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <openMVG/sfm/pipelines/sequential/sequential_SfM.hpp>

#include "BundleAdjustmentAlgorithm.h"
#include "ConfigNodes.h"
#include "JsonConfig.h"
#include "ProcessingData.h"
#include "ReconstructionParams.h"
#include "Scene.h"
#include "PathUtils.h"
#include "Logger.h"

static openMVG::cameras::Intrinsic_Parameter_Type GetIntrinsicParameterType(bool constantDistortionParams, bool constantFocalLength, bool constantPrincipalPoint)
{
    if(constantPrincipalPoint && constantFocalLength && constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::NONE;
    }

    if(!constantPrincipalPoint && constantFocalLength && constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_PRINCIPAL_POINT;
    }

    if(constantPrincipalPoint && !constantFocalLength && constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_FOCAL_LENGTH;
    }

    if(constantPrincipalPoint && constantFocalLength && !constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_DISTORTION;
    }

    if(!constantPrincipalPoint && !constantFocalLength && constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_PRINCIPAL_POINT | openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_FOCAL_LENGTH;
    }

    if(!constantPrincipalPoint && constantFocalLength && !constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_PRINCIPAL_POINT | openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_DISTORTION;
    }

    if(constantPrincipalPoint && !constantFocalLength && !constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_FOCAL_LENGTH | openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_DISTORTION;
    }

    if(!constantPrincipalPoint && !constantFocalLength && !constantDistortionParams)
    {
        return openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_ALL;
    }

    return openMVG::cameras::Intrinsic_Parameter_Type::NONE;
}

static openMVG::ETriangulationMethod GetTriangulationMethod(const std::string& triangulationMethod)
{
    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::DirectLinearTransformTriangulationMethod)
    {
        return openMVG::ETriangulationMethod::DIRECT_LINEAR_TRANSFORM;
    }

    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::L1AngularTriangulationMethod)
    {
        return openMVG::ETriangulationMethod::L1_ANGULAR;
    }

    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::LInfinityAngularTriangulationMethod)
    {
        return openMVG::ETriangulationMethod::LINFINITY_ANGULAR;
    }

    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::InverseDepthWeightedMidpointTriangulationMethod)
    {
        return openMVG::ETriangulationMethod::INVERSE_DEPTH_WEIGHTED_MIDPOINT;
    }

    return openMVG::ETriangulationMethod::DEFAULT;
}

static openMVG::resection::SolverType GetResectionMethod(const std::string& resectionMethod)
{
    if (resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::DirectLinearTransform6Points)
    {
        return openMVG::resection::SolverType::DLT_6POINTS;
    }

    if (resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::P3P_KE_CVPR17)
    {
        return openMVG::resection::SolverType::P3P_KE_CVPR17;
    }

    if (resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::P3P_KNEIP_CVPR11)
    {
        return openMVG::resection::SolverType::P3P_KNEIP_CVPR11;
    }

    if (resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::P3P_NORDBERG_ECCV18)
    {
        return openMVG::resection::SolverType::P3P_NORDBERG_ECCV18;
    }

    if (resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UP2P_KUKELOVA_ACCV10)
    {
        return openMVG::resection::SolverType::UP2P_KUKELOVA_ACCV10;
    }

    return openMVG::resection::SolverType::DEFAULT;
}

namespace Algorithms
{

BundleAdjustmentAlgorithm::BundleAdjustmentAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
isInitialized_(false),
useMotionPrior_(false),
constantDistortionParams_(false),
constantFocalLength_(false),
constantPrincipalPoint_(false),
saveResult_(false)
{
    InitializeInternal(config);
}

bool BundleAdjustmentAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    using namespace Log;
    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();
    auto& reconstructionParams = dataset->GetReconstructionParams();
    auto& sfmData = *reconstructionParams->sfMData_;
    auto& featuresProviderPointer = reconstructionParams->featuresProvider_;
    auto& matchesProviderPointer = reconstructionParams->matchesProvider_;

    std::filesystem::path modelsPath = Utils::PathUtils::GetExecutableFolderPath();
    modelsPath /= Config::ConfigNodes::ProjectSubdirectories::Models;

    if (!std::filesystem::exists(modelsPath))
    {
        std::error_code errorCode;
        std::filesystem::create_directories(modelsPath, errorCode);
        if(errorCode)
        {
            for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
            << "Failed to create model's directory " << modelsPath << ". Error code "
            << errorCode.value() << ": " << errorCode.message();

            return false;
        }
    }

    auto reconstructionEngine = std::make_unique<openMVG::sfm::SequentialSfMReconstructionEngine>(sfmData, (modelsPath / datasetUUID).string());

    auto intrinsicParameterType = GetIntrinsicParameterType(constantDistortionParams_, constantFocalLength_, constantPrincipalPoint_);
    auto triangulationMethod = GetTriangulationMethod(triangulationMethod_);
    auto resectionMethod = GetResectionMethod(resectionMethod_);

    reconstructionEngine->SetMatchesProvider(matchesProviderPointer.get());
    reconstructionEngine->SetFeaturesProvider(featuresProviderPointer.get());
    reconstructionEngine->Set_Intrinsics_Refinement_Type(intrinsicParameterType);
    reconstructionEngine->SetUnknownCameraType(openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_RADIAL3);
    reconstructionEngine->Set_Use_Motion_Prior(useMotionPrior_);
    reconstructionEngine->SetTriangulationMethod(triangulationMethod);
    reconstructionEngine->SetResectionMethod(resectionMethod);

    if(reconstructionEngine->Process())
    {
        auto& newSfmData = reconstructionEngine->Get_SfM_Data();
        auto& scenePointer = reconstructionParams->scene_;
        scenePointer = std::make_shared<Scene>(newSfmData, *dataset, 0);
        reconstructionParams->matchesProvider_ = nullptr;
        reconstructionParams->regionsProvider_ = nullptr;
        reconstructionParams->matches_ = nullptr;
        reconstructionParams->featuresProvider_ = nullptr;
        reconstructionParams->sfMData_ = nullptr;

        if(saveResult_)
        {
            scenePointer->Save("scene.mvs");
            scenePointer->pointcloud.Save("scene.ply");
        }

        return true;
    }
    else
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Failed to perform bundle adjustment algorithm on dataset " << datasetUUID << ".";
        return false;
    }
}

void BundleAdjustmentAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

bool BundleAdjustmentAlgorithm::ValidateResectionMethod(const std::string& resectionMethod)
{
    if(resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::DirectLinearTransform6Points)
    {
        return true;
    }

    if(resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::P3P_KE_CVPR17)
    {
        return true;
    }

    if(resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::P3P_KNEIP_CVPR11)
    {
        return true;
    }

    if(resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::P3P_NORDBERG_ECCV18)
    {
        return true;
    }

    if(resectionMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UP2P_KUKELOVA_ACCV10)
    {
        return true;
    }

    return false;
}

bool BundleAdjustmentAlgorithm::ValidateTriangulationMethod(const std::string& triangulationMethod)
{
    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::DirectLinearTransformTriangulationMethod)
    {
        return true;
    }

    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::L1AngularTriangulationMethod)
    {
        return true;
    }

    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::LInfinityAngularTriangulationMethod)
    {
        return true;
    }

    if(triangulationMethod == Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::InverseDepthWeightedMidpointTriangulationMethod)
    {
        return true;
    }

    return false;
}

void BundleAdjustmentAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;
    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseMotionPrior))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseMotionPrior
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantDistortionParams))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantDistortionParams
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantFocalLength))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantFocalLength
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantPrincipalPoint))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantPrincipalPoint
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::ResectionMethod))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::ResectionMethod
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::TriangulationMethod))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::TriangulationMethod
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::SaveResult))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
                    << "Invalid bundle adjustment algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::SaveResult
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    auto resectionMethod = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::ResectionMethod]->ToString();
    boost::algorithm::trim(resectionMethod);
    boost::algorithm::to_upper(resectionMethod);

    if(!ValidateResectionMethod(resectionMethod))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. Unknown resection method.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    auto triangulationMethod = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::TriangulationMethod]->ToString();
    boost::algorithm::trim(triangulationMethod);
    boost::algorithm::to_upper(triangulationMethod);

    if(!ValidateTriangulationMethod(triangulationMethod))
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
        << "Invalid bundle adjustment algorithm configuration. Unknown triangulation method.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void BundleAdjustmentAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    using namespace Log;
    if(!isInitialized_)
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
        << "Initializing bundle adjustment algorithm ...";
        ValidateConfig(config);

        constantPrincipalPoint_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantPrincipalPoint]->ToBool();
        constantFocalLength_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantFocalLength]->ToBool();
        constantDistortionParams_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseConstantDistortionParams]->ToBool();

        useMotionPrior_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::UseMotionPrior]->ToBool();

        triangulationMethod_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::TriangulationMethod]->ToString();
        resectionMethod_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::ResectionMethod]->ToString();

        saveResult_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::BundleAdjustmentAlgorithm::SaveResult]->ToBool();

        isInitialized_ = true;
    }
    else
    {
        for (Logger::RecordStream stream(Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(SEVERITY_LEVEL::WARNING, __PRETTY_FUNCTION__)
        << "Bundle adjustment algorithm is already initialized.";
    }
}


}