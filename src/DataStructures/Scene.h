#ifndef SCENE_H
#define SCENE_H


#include <OpenMVS/ConfigLocal.h>
#include <OpenMVS/MVS/Common.h>
#include <OpenMVS/MVS/Scene.h>

class DenseDepthMapData;

namespace openMVG::sfm
{
    struct SfM_Data;
}

namespace DataStructures
{
    class ModelDataset;
}

class Scene : public MVS::Scene
{
public:
    Scene();
    Scene(const openMVG::sfm::SfM_Data &sfm_data, DataStructures::ModelDataset& modelDataset, unsigned _nMaxThreads);

    void Release();
    bool IsEmpty() const;

    bool LoadInterface(const String& fileName);
    bool SaveInterface(const String& fileName, int version=-1) const;

    bool LoadDMAP(const String& fileName);
    bool Import(const String& fileName);

    bool Save(const String& fileName, ARCHIVE_TYPE type=ARCHIVE_BINARY_ZIP) const;

    bool SelectNeighborViews(uint32_t ID, IndexArr& points, unsigned nMinViews=3, unsigned nMinPointViews=2, float fOptimAngle=FD2R(10));
    static bool FilterNeighborViews(MVS::ViewScoreArr& neighbors, float fMinArea=0.1f, float fMinScale=0.2f, float fMaxScale=2.4f, float fMinAngle=FD2R(3), float fMaxAngle=FD2R(45), unsigned nMaxViews=12);

    // Dense reconstruction
    bool DenseReconstruction(int nFusionMode=0);
    bool ComputeDepthMaps(DenseDepthMapData& data);
    void DenseReconstructionEstimate(DenseDepthMapData& data);
    void DenseReconstructionFilter(DenseDepthMapData& data);


    bool RefineMeshCUDA(unsigned nResolutionLevel,
                        unsigned nMinResolution,
                        unsigned nMaxViews,
                        float fDecimateMesh,
                        unsigned nCloseHoles,
                        unsigned nEnsureEdgeSize,
                        unsigned nMaxFaceArea,
                        unsigned nScales,
                        float fScaleStep,
                        unsigned nAlternatePair,
                        float fRegularityWeight,
                        float fRatioRigidityElasticity,
                        float fGradientStep);


    // Mesh reconstruction
    bool ReconstructMesh(float distInsert=2, bool bUseFreeSpaceSupport=true, unsigned nItersFixNonManifold=4,
                         float kSigma=2.f, float kQual=1.f, float kb=4.f,
                         float kf=3.f, float kRel=0.1f/*max 0.3*/, float kAbs=1000.f/*min 500*/, float kOutl=400.f/*max 700.f*/,
                         float kInf=(float)(INT_MAX/8));
  // Mesh texturing
    bool TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, float fOutlierThreshold=0.f, float fRatioDataSmoothness=0.3f, bool bGlobalSeamLeveling=true, bool bLocalSeamLeveling=true, unsigned nTextureSizeMultiple=0, unsigned nRectPackingHeuristic=3, Pixel8U colEmpty=Pixel8U(255,127,39));
};

#endif // SCENE_H
