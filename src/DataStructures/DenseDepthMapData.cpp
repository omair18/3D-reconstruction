#include "Scene.h"
#include "DenseDepthMapData.h"



DenseDepthMapData::DenseDepthMapData(Scene &_scene, int _nFusionMode)
    : scene(_scene), depthMaps(_scene), idxImage(0), sem(1), nFusionMode(_nFusionMode)
{
    if (nFusionMode < 0)
    {
        MVS::STEREO::SemiGlobalMatcher::CreateThreads(scene.nMaxThreads);
        if (nFusionMode == -1)
            MVS::OPTDENSE::nOptimize &= ~MVS::OPTDENSE::OPTIMIZE;
    }
}

DenseDepthMapData::~DenseDepthMapData()
{
    if (nFusionMode < 0)
        MVS::STEREO::SemiGlobalMatcher::DestroyThreads();
}

void DenseDepthMapData::SignalCompleteDepthmapFilter()
{
    if (Thread::safeDec(idxImage) == 0)
        sem.Signal((unsigned)images.GetSize()*2);
}
