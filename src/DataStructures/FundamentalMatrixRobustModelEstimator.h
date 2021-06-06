#ifndef FUNDAMENTAL_MATRIX_ROBUST_MODEL_ESTIMATOR
#define FUNDAMENTAL_MATRIX_ROBUST_MODEL_ESTIMATOR

#include <memory>
#include <cstdint>
#include <openMVG/types.hpp>
#include <openMVG/matching/indMatch.hpp>

namespace openMVG
{
    namespace sfm
    {
        struct SfM_Data;
    }
}

class RegionsProvider;

class FundamentalMatrixRobustModelEstimator
{
    float m_dPrecision;    // upper_bound precision used for robust estimation
    unsigned int m_stIteration; // maximal number of iteration for robust estimation
    //
    //-- Stored data
    Eigen::Matrix<double, 3, 3> m_F;
public:
    FundamentalMatrixRobustModelEstimator(float dPrecision, unsigned int iteration) :
    m_dPrecision(dPrecision),
    m_stIteration(iteration),
    m_F(Eigen::Matrix<double, 3, 3>::Identity())
    {

    }

    bool robust_estimation(const openMVG::sfm::SfM_Data *sfm_data,
                           const std::shared_ptr<RegionsProvider>& regions_provider,
                           const openMVG::Pair& pairIndex,
                           const openMVG::matching::IndMatches& vec_PutativeMatches,
                           openMVG::matching::IndMatches& geometric_inliers);

};

#endif
