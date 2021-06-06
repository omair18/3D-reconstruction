#ifndef DENSE_DEPTH_MAP_DATA_H
#define DENSE_DEPTH_MAP_DATA_H

#include "DepthMapsData.h"

class Scene;

class DenseDepthMapData
{
public:
    Scene& scene;
    MVS::IIndexArr images;
    MVS::IIndexArr neighborsMap;
    DepthMapsData depthMaps;
    volatile Thread::safe_t idxImage;
    SEACAVE::EventQueue events; // internal events queue (processed by the working threads)
    Semaphore sem;
    CAutoPtr<Util::Progress> progress;
    int nFusionMode;
    MVS::STEREO::SemiGlobalMatcher sgm;

    DenseDepthMapData(Scene& _scene, int _nFusionMode=0);
    ~DenseDepthMapData();

    void SignalCompleteDepthmapFilter();

};

#endif
