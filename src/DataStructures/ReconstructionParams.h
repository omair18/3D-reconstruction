#ifndef RECONSTRUCTION_PARAMS_H
#define RECONSTRUCTION_PARAMS_H

#include <openMVG/cameras/cameras.hpp>
#include <openMVG/sfm/sfm.hpp>

/**
 * @namespace DataStructures
 *
 * @brief Namespace of libdatastructures library.
 */
namespace DataStructures
{

struct ReconstructionParams
{
    openMVG::cameras::EINTRINSIC distortionFunction_;
    openMVG::sfm::SfM_Data sfMData_;
};

}

#endif // RECONSTRUCTION_PARAMS_H
