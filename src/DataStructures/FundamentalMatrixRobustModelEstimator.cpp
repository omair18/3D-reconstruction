#include "FundamentalMatrixRobustModelEstimator.h"
#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/multiview/solver_fundamental_kernel.hpp>
#include <openMVG/matching_image_collection/Geometric_Filter_utils.hpp>
#include <openMVG/robust_estimation/robust_estimator_ACRansacKernelAdaptator.hpp>
#include <openMVG/robust_estimation/robust_estimator_ACRansac.hpp>
#include "RegionsProvider.h"

bool FundamentalMatrixRobustModelEstimator::robust_estimation(const openMVG::sfm::SfM_Data *sfm_data,
                                                              const std::shared_ptr<RegionsProvider>& regions_provider,
                                                              const openMVG::Pair& pairIndex,
                                                              const openMVG::matching::IndMatches& vec_PutativeMatches,
                                                              openMVG::matching::IndMatches& geometric_inliers)
{
    geometric_inliers.clear();

    // Get back corresponding view index
    const openMVG::IndexT iIndex = pairIndex.first;
    const openMVG::IndexT jIndex = pairIndex.second;


    openMVG::Mat2X xI,xJ;
    openMVG::matching_image_collection::MatchesPairToMat(pairIndex, vec_PutativeMatches, sfm_data, regions_provider, xI, xJ);


    const openMVG::robust::ACKernelAdaptor<openMVG::fundamental::kernel::SevenPointSolver, openMVG::fundamental::kernel::EpipolarDistanceError,
            openMVG::UnnormalizerT, Eigen::Matrix<double, 3, 3>>
            kernel(
                xI, sfm_data->GetViews().at(iIndex)->ui_width, sfm_data->GetViews().at(iIndex)->ui_height,
                xJ, sfm_data->GetViews().at(jIndex)->ui_width, sfm_data->GetViews().at(jIndex)->ui_height, true);

    // Robustly estimate the Fundamental matrix with A Contrario ransac
    const double upper_bound_precision = m_dPrecision * m_dPrecision;
    std::vector<uint32_t> vec_inliers;
    const std::pair<double,double> ACRansacOut =
            openMVG::robust:: ACRANSAC(kernel, vec_inliers, m_stIteration, &m_F, upper_bound_precision);

    if (vec_inliers.size()
            >
            openMVG::robust::ACKernelAdaptor<openMVG::fundamental::kernel::SevenPointSolver, openMVG::fundamental::kernel::EpipolarDistanceError, openMVG::UnnormalizerT,
            Eigen::Matrix<double, 3, 3>>::MINIMUM_SAMPLES *2.5
            )
    {
        // update geometric_inliers
        geometric_inliers.reserve(vec_inliers.size());
        for (const uint32_t & index : vec_inliers)
        {
            geometric_inliers.push_back( vec_PutativeMatches[index] );
        }
        return true;
    }
    else
    {
        vec_inliers.clear();
        return false;
    }
}
