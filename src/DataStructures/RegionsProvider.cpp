#include "RegionsProvider.h"


std::shared_ptr<openMVG::features::Regions> RegionsProvider::get(const openMVG::IndexT x) const
{
    return openMVG::sfm::Regions_Provider::get(x);
}

bool RegionsProvider::load(const openMVG::sfm::SfM_Data &sfm_data, const std::string &feat_directory, std::unique_ptr<openMVG::features::Regions> &region_type, C_Progress *my_progress_bar)
{
    return openMVG::sfm::Regions_Provider::load(sfm_data, feat_directory, region_type, my_progress_bar);
}

openMVG::Hash_Map<openMVG::IndexT, std::shared_ptr<openMVG::features::Regions> > &RegionsProvider::get_regions_map()
{
    return cache_;
}

std::unique_ptr<openMVG::features::Regions>& RegionsProvider::get_regions_type()
{
    return region_type_;
}
