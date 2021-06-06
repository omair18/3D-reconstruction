#include <openMVG/matching/indMatch.hpp>
#include <openMVG/sfm/pipelines/sfm_matches_provider.hpp>

#include "KeyPointFilteringAlgorithm.h"
#include "ConfigNodes.h"
#include "JsonConfig.h"
#include "ProcessingData.h"
#include "ModelDataset.h"
#include "ReconstructionParams.h"
#include "FundamentalMatrixRobustModelEstimator.h"
#include "Logger.h"

namespace Algorithms
{

KeyPointFilteringAlgorithm::KeyPointFilteringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
isInitialized_(false),
maxIterations_(2024),
estimationPrecision_(4.0f)
{
    InitializeInternal(config);
}

KeyPointFilteringAlgorithm::~KeyPointFilteringAlgorithm() = default;

bool KeyPointFilteringAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();
    auto& reconstructionParams = dataset->GetReconstructionParams();
    auto& matchesMap = *reconstructionParams->matches_;
    auto& sfmDataPointer = reconstructionParams->sfMData_;
    auto& regionsProvider = reconstructionParams->regionsProvider_;
    auto& matchesProvider = *reconstructionParams->matchesProvider_;

    openMVG::matching::PairWiseMatches geometricMatchesMap;

    for (const auto& [pair, matches] : matchesMap)
    {
        openMVG::matching::IndMatches putativeInliers;
        if (modelEstimator_->robust_estimation(sfmDataPointer.get(), regionsProvider, pair, matches, putativeInliers))
        {
            geometricMatchesMap.insert( {pair, std::move(putativeInliers)});
        }
    }

    matchesProvider.pairWise_matches_ = std::move(geometricMatchesMap);

    return true;
}

void KeyPointFilteringAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void KeyPointFilteringAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::KeyPointFilteringAlgorithm::MaxIterations))
    {
        LOG_ERROR() << "Invalid key point filtering algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::KeyPointFilteringAlgorithm::MaxIterations
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if (!config->Contains(Config::ConfigNodes::AlgorithmsConfig::KeyPointFilteringAlgorithm::EstimationPrecision))
    {
        LOG_ERROR() << "Invalid key point filtering algorithm configuration. There must be "
                    << Config::ConfigNodes::AlgorithmsConfig::KeyPointFilteringAlgorithm::EstimationPrecision
                    << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void KeyPointFilteringAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(!isInitialized_)
    {
        LOG_TRACE() << "Initializing key point filtering algorithm ...";

        ValidateConfig(config);

        maxIterations_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::KeyPointFilteringAlgorithm::MaxIterations]->ToInt32();
        estimationPrecision_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::KeyPointFilteringAlgorithm::EstimationPrecision]->ToFloat();

        modelEstimator_ = std::make_unique<FundamentalMatrixRobustModelEstimator>(estimationPrecision_, maxIterations_);

        LOG_TRACE() << "Key point filtering algorithm was successfully initialized.";
        isInitialized_ = true;
    }
    else
    {
        LOG_WARNING() << "Key point filtering algorithm is already initialized";
    }
}

}
