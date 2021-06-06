#include <openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp>
#include <openMVG/matching_image_collection/Pair_Builder.hpp>

#include "KeyPointMatchingAlgorithm.h"
#include "ProcessingData.h"
#include "ReconstructionParams.h"
#include "JsonConfig.h"
#include "RegionsProvider.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace Algorithms
{

KeyPointMatchingAlgorithm::KeyPointMatchingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
distanceRatio_(0),
matcher_(nullptr)
{
    InitializeInternal(config);
}

KeyPointMatchingAlgorithm::~KeyPointMatchingAlgorithm() = default;

bool KeyPointMatchingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();
    auto& imageDescriptors = dataset->GetImagesDescriptors();
    auto& reconstructionParams = dataset->GetReconstructionParams();
    auto& regionsProvider = reconstructionParams->regionsProvider_;
    auto& sfmData = *reconstructionParams->sfMData_;
    auto& matches = *reconstructionParams->matches_;

    LOG_TRACE() << "Matching key points of dataset " << datasetUUID << " ...";

    if(!regionsProvider->get_regions_type()->IsScalar())
    {
        LOG_ERROR() << "Unsupported regions type";
        return false;
    }

    openMVG::Pair_Set pairs = openMVG::exhaustivePairs(sfmData.GetViews().size());

    matcher_->Match(regionsProvider, pairs, matches);

    return true;
}

void KeyPointMatchingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void KeyPointMatchingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::KeyPointMatchingAlgorithm::DistanceRatio))
    {
        LOG_ERROR() << "Invalid key point matching algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::KeyPointMatchingAlgorithm::DistanceRatio
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void KeyPointMatchingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(!matcher_)
    {
        LOG_TRACE() << "Initializing key point matching algorithm ...";
        ValidateConfig(config);

        distanceRatio_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::KeyPointMatchingAlgorithm::DistanceRatio]->ToFloat();

        matcher_ = std::make_unique<openMVG::matching_image_collection::Cascade_Hashing_Matcher_Regions>(distanceRatio_);
    }
    else
    {
        LOG_WARNING() << "Key point matching algorithm was already initialized.";
    }
}

}