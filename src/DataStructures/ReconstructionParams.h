/**
 * @file ReconstructionParams.h
 *
 * @brief
 */

#ifndef RECONSTRUCTION_PARAMS_H
#define RECONSTRUCTION_PARAMS_H

#include <memory>

namespace openMVG
{
    namespace sfm
    {
        struct SfM_Data;
        struct Features_Provider;
        struct Matches_Provider;
    }
}

class RegionsProvider;

/**
 * @namespace DataStructures
 *
 * @brief Namespace of libdatastructures library.
 */
namespace DataStructures
{

class ReconstructionParams
{
public:

    ReconstructionParams();

    ~ReconstructionParams();

    std::shared_ptr<openMVG::sfm::SfM_Data> sfMData_;

    std::shared_ptr<openMVG::sfm::Features_Provider> featuresProvider_;

    std::shared_ptr<openMVG::sfm::Matches_Provider> matchesProvider_;

    std::shared_ptr<RegionsProvider> regionsProvider_;

};

}

#endif // RECONSTRUCTION_PARAMS_H
