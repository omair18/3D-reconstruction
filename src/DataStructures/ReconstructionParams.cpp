#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/sfm/pipelines/sfm_features_provider.hpp>
#include <openMVG/sfm/pipelines/sfm_matches_provider.hpp>
#include <openMVG/matching/indMatch_io.hpp>

#include "ReconstructionParams.h"
#include "RegionsProvider.h"
#include "Scene.h"

DataStructures::ReconstructionParams::ReconstructionParams() :
sfMData_(std::make_shared<openMVG::sfm::SfM_Data>()),
regionsProvider_(std::make_shared<RegionsProvider>()),
featuresProvider_(std::make_shared<openMVG::sfm::Features_Provider>()),
matchesProvider_(std::make_shared<openMVG::sfm::Matches_Provider>()),
matches_(std::make_shared<openMVG::matching::PairWiseMatches>()),
scene_(nullptr)
{

}

DataStructures::ReconstructionParams::~ReconstructionParams() = default;


