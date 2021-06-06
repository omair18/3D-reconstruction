#ifndef DEPTH_MAPS_DATA_H
#define DEPTH_MAPS_DATA_H

#include <OpenMVS/MVS.h>
#include <OpenMVS/MVS/DepthMap.h>

class Scene;

class DepthMapsData
{

public:
    DepthMapsData(Scene& _scene);
    ~DepthMapsData();

    bool SelectViews(MVS::IIndexArr& images, MVS::IIndexArr& imagesMap, MVS::IIndexArr& neighborsMap);
    bool SelectViews(MVS::DepthData& depthData);
    bool InitViews(MVS::DepthData& depthData, MVS::IIndex idxNeighbor, MVS::IIndex numNeighbors);
    bool InitDepthMap(MVS::DepthData& depthData);
    bool EstimateDepthMap(MVS::IIndex idxImage);

    bool RemoveSmallSegments(MVS::DepthData& depthData);
    bool GapInterpolation(MVS::DepthData& depthData);

    bool FilterDepthMap(MVS::DepthData& depthData, const MVS::IIndexArr& idxNeighbors, bool bAdjust=true);
    void FuseDepthMaps(MVS::PointCloud& pointcloud, bool bEstimateColor, bool bEstimateNormal);

    Scene& scene;

    MVS::DepthDataArr arrDepthData;

    // used internally to estimate the depth-maps
    Image8U::Size prevDepthMapSize; // remember the size of the last estimated depth-map
    Image8U::Size prevDepthMapSizeTrg; // ... same for target image
    MVS::DepthEstimator::MapRefArr coords; // map pixel index to zigzag matrix coordinates
    MVS::DepthEstimator::MapRefArr coordsTrg; // ... same for target image
};

#endif
