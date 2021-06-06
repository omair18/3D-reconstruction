#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_features_provider.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "AKAZEKeypointDetectionAlgorithm.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "ProcessingData.h"
#include "ReconstructionParams.h"
#include "ImageDescriptor.h"
#include "RegionsProvider.h"
#include "Logger.h"

const static std::unordered_map<const std::string, cv::KAZE::DiffusivityType, boost::hash<const std::string>> diffusionFunctions =
        {
                { Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::PeronaMalikG1DiffusionFunction, cv::KAZE::DiffusivityType::DIFF_PM_G1 },
                { Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::PeronaMalikG2DiffusionFunction, cv::KAZE::DiffusivityType::DIFF_PM_G2 },
                { Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::WeickertDiffusionFunction, cv::KAZE::DiffusivityType::DIFF_WEICKERT },
                { Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::CharbonnierDiffusionFunction, cv::KAZE::DiffusivityType::DIFF_CHARBONNIER },
        };

namespace Algorithms
{

AKAZEKeypointDetectionAlgorithm::AKAZEKeypointDetectionAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
threshold_(0.f),
octaves_(0),
sublayersPerOctave_(0),
anisotropicDiffusionFunction_(""),
buffer_(),
akaze_(nullptr)
{
    InitializeInternal(config);
}

AKAZEKeypointDetectionAlgorithm::~AKAZEKeypointDetectionAlgorithm() = default;

bool AKAZEKeypointDetectionAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();
    auto& imageDescriptors = dataset->GetImagesDescriptors();
    auto& reconstructionParams = dataset->GetReconstructionParams();

    auto& sfmData = *reconstructionParams->sfMData_;
    auto& regionsProvider = *reconstructionParams->regionsProvider_;
    auto& featuresProvider = *reconstructionParams->featuresProvider_;

    auto& imageRegions = regionsProvider.get_regions_map();
    auto& regionsFeatures = featuresProvider.feats_per_view;

    LOG_TRACE() << "Detecting key points of dataset with ID " << datasetUUID;

    for(int i = 0; i < static_cast<int>(imageDescriptors.size()); ++i)
    {
        LOG_TRACE() << "Detecting key points on image " << i << "/" << imageDescriptors.size() << " ...";
        auto& view = sfmData.views[i];
        std::vector<cv::KeyPoint> keyPoints;
        auto& imageDescriptor = imageDescriptors[i];
        auto regions = std::make_unique<openMVG::features::AKAZE_Float_Regions>();

        if(imageDescriptor.GetDataLocation() == DataStructures::ImageDescriptor::LOCATION::HOST)
        {
            auto& image = *imageDescriptor.GetHostImage();
            cv::Mat descriptorsMatrix;
            cv::cvtColor(image, buffer_, cv::COLOR_BGR2GRAY);
            akaze_->detectAndCompute(buffer_, cv::Mat(), keyPoints, descriptorsMatrix);
            if(!keyPoints.empty())
            {
                regions->Features().reserve(keyPoints.size());
                regions->Descriptors().reserve(keyPoints.size());

                openMVG::features::Descriptor<float, 64> descriptor;
                int cpt = 0;
                for (auto i_keypoint = keyPoints.begin(); i_keypoint != keyPoints.end(); ++i_keypoint, ++cpt)
                {
                    regions->Features().emplace_back(openMVG::features::SIOPointFeature((*i_keypoint).pt.x, (*i_keypoint).pt.y, (*i_keypoint).size, (*i_keypoint).angle));
                    memcpy(descriptor.data(),
                           descriptorsMatrix.ptr<typename openMVG::features::Descriptor<float, 64>::bin_type>(cpt),
                           openMVG::features::Descriptor<float, 64>::static_size*sizeof(openMVG::features::Descriptor<float, 64>::bin_type));
                    regions->Descriptors().emplace_back(descriptor);
                }
            }

            regionsFeatures[view->id_view] = regions->GetRegionsPositions();
            imageRegions[view->id_view] = std::move(regions);
        }
        else
        {
            LOG_ERROR() << "Invalid data location.";
            return false;
        }
    }

    regionsProvider.get_regions_type() = std::make_unique<openMVG::features::AKAZE_Float_Regions>();

    return true;
}

void AKAZEKeypointDetectionAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void AKAZEKeypointDetectionAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::Threshold))
    {
        LOG_ERROR() << "Invalid AKAZE image key point detection algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::Threshold
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::Octaves))
    {
        LOG_ERROR() << "Invalid AKAZE image key point detection algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::Octaves
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::SublayersPerOctave))
    {
        LOG_ERROR() << "Invalid AKAZE image key point detection algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::SublayersPerOctave
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::AnisotropicDiffusionFunction))
    {
        LOG_ERROR() << "Invalid AKAZE image key point detection algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::AnisotropicDiffusionFunction
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    auto diffusionFunction = (*config)[Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::AnisotropicDiffusionFunction]->ToString();

    boost::algorithm::trim(diffusionFunction);
    boost::algorithm::to_upper(diffusionFunction);

    if (diffusionFunction == Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::PeronaMalikG1DiffusionFunction)
        return;

    if (diffusionFunction == Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::PeronaMalikG2DiffusionFunction)
        return;

    if (diffusionFunction == Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::WeickertDiffusionFunction)
        return;

    if (diffusionFunction == Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::CharbonnierDiffusionFunction)
        return;

    LOG_ERROR() << "Invalid AKAZE image key point detection algorithm configuration. Unknown diffusion function.";
    throw std::runtime_error("Invalid algorithm configuration.");
}

void AKAZEKeypointDetectionAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(!akaze_)
    {
        LOG_TRACE() << "Initializing AKAZE key point detection algorithm ...";

        ValidateConfig(config);

        octaves_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::Octaves]->ToInt32();
        LOG_TRACE() << "AKAZE image key point detection algorithm's octaves amount was set to " << octaves_;

        sublayersPerOctave_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::SublayersPerOctave]->ToInt32();
        LOG_TRACE() << "AKAZE image key point detection algorithm's amount of sublayers per octave was set to " << sublayersPerOctave_;

        threshold_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::Threshold]->ToFloat();
        LOG_TRACE() << "AKAZE image key point detection algorithm's amount of sublayers per octave was set to " << sublayersPerOctave_;

        anisotropicDiffusionFunction_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::AKAZEKeyPointDetectionAlgorithm::AnisotropicDiffusionFunction]->ToString();
        LOG_TRACE() << "AKAZE image key point detection algorithm's diffusion function was set to " << anisotropicDiffusionFunction_;
        akaze_ = cv::AKAZE::create(cv::AKAZE::DescriptorType::DESCRIPTOR_KAZE, 0, 1, threshold_, octaves_, sublayersPerOctave_,
                                   diffusionFunctions.at(anisotropicDiffusionFunction_));

        LOG_TRACE() << "AKAZE image key point detection algorithm was successfully initialized";
    }
    else
    {
        LOG_WARNING() << "AKAZE key point detection algorithm was already initialized.";
    }

}

}
