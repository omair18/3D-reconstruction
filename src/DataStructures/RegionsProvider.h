#ifndef REGIONS_PROVIDER
#define REGIONS_PROVIDER

#include <openMVG/sfm/pipelines/sfm_regions_provider.hpp>

class RegionsProvider : public openMVG::sfm::Regions_Provider
{

public:

    std::shared_ptr<openMVG::features::Regions> get(const openMVG::IndexT x) const override;
    bool load(const openMVG::sfm::SfM_Data &sfm_data, const std::string &feat_directory, std::unique_ptr<openMVG::features::Regions> &region_type, C_Progress *my_progress_bar = nullptr) override;
    openMVG::Hash_Map<openMVG::IndexT, std::shared_ptr<openMVG::features::Regions>>& get_regions_map();
    std::unique_ptr<openMVG::features::Regions>& get_regions_type();
};

#endif
