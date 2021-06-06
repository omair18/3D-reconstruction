#include <openMVG/sfm/sfm.hpp>
#include <openMVG/cameras/Camera_Pinhole.hpp>
#include <OpenMVS/Math/TRWS/MRFEnergy.h>

#include "Scene.h"
#include <OpenMVS/Common/Common.h>
#include "DenseDepthMapData.h"
#include "DepthMapsData.h"
#include "ModelDataset.h"
#include "ImageDescriptor.h"


enum EVENT_TYPE {
    EVT_FAIL = 0,
    EVT_CLOSE,

    EVT_PROCESSIMAGE,

    EVT_ESTIMATEDEPTHMAP,
    EVT_OPTIMIZEDEPTHMAP,

    EVT_FILTERDEPTHMAP
};

class EVTFail : public Event
{
public:
    EVTFail() : Event(EVT_FAIL) {}
};
class EVTClose : public Event
{
public:
    EVTClose() : Event(EVT_CLOSE) {}
};

class EVTProcessImage : public Event
{
public:
    MVS::IIndex idxImage;
    EVTProcessImage(MVS::IIndex _idxImage) : Event(EVT_PROCESSIMAGE), idxImage(_idxImage) {}
};

class EVTEstimateDepthMap : public Event
{
public:
    MVS::IIndex idxImage;
    EVTEstimateDepthMap(MVS::IIndex _idxImage) : Event(EVT_ESTIMATEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTOptimizeDepthMap : public Event
{
public:
    MVS::IIndex idxImage;
    EVTOptimizeDepthMap(MVS::IIndex _idxImage) : Event(EVT_OPTIMIZEDEPTHMAP), idxImage(_idxImage) {}
};

class EVTFilterDepthMap : public Event
{
public:
    MVS::IIndex idxImage;
    EVTFilterDepthMap(MVS::IIndex _idxImage) : Event(EVT_FILTERDEPTHMAP), idxImage(_idxImage) {}
};

Scene::Scene(const openMVG::sfm::SfM_Data &sfm_data, DataStructures::ModelDataset& dataset, unsigned _nMaxThreads)
    : MVS::Scene(_nMaxThreads)
{
    auto& imageDescriptors = dataset.GetImagesDescriptors();
    nCalibratedImages = 0;
    std::unordered_map<openMVG::IndexT, uint32_t> map_intrinsic, map_view;
    for(const auto& intrinsic : sfm_data.GetIntrinsics())
    {
        const auto * cam = dynamic_cast<const openMVG::cameras::Pinhole_Intrinsic*>(intrinsic.second.get());
        if (map_intrinsic.count(intrinsic.first) == 0)
            map_intrinsic.insert(std::make_pair(intrinsic.first, platforms.size()));
        MVS::Platform platform;
        // add the camera
        MVS::Platform::Camera camera;
        //camera.K = cam->K();
        for (int i = 0; i < 9; i++)
        {
            camera.K.val[i] = cam->K().data()[i];
        }
        // sub-pose
        camera.R = Eigen::Matrix<double, 3, 3>::Identity();
        camera.C = Eigen::Vector3d::Zero();
        platform.cameras.push_back(std::move(camera));
        platforms.push_back(std::move(platform));
    }

    images.reserve(sfm_data.views.size());
    for (const auto& view : sfm_data.GetViews())
    {
        if (sfm_data.IsPoseAndIntrinsicDefined(view.second.get()))
        {
            map_view[view.first] = images.size();
            MVS::Image image;
            image.name = view.second->s_Img_path;
            image.platformID = map_intrinsic.at(view.second->id_intrinsic);
            MVS::Platform& platform = platforms[image.platformID];
            image.cameraID = 0;
            image.width = view.second->ui_width;
            image.height = view.second->ui_height;
            image.scale = 1.f;

            auto& source_image = *imageDescriptors[std::stoi(view.second->s_Img_path)].GetHostImage();

            image.image.cols = source_image.cols;
            image.image.rows = source_image.rows;
            image.image.step[0] = source_image.step[0];
            image.image.step[1] = source_image.step[1];
            image.image.dims = source_image.dims;
            image.image.datastart = source_image.datastart;
            image.image.data = source_image.data;
            image.image.dataend = source_image.dataend;
            source_image.data = nullptr;
            source_image.datastart = nullptr;
            source_image.dataend = nullptr;
            image.image.flags = source_image.flags;
            source_image.flags = 0;
            source_image.cols = 0;
            source_image.rows = 0;
            source_image.dims = 0;

            MVS::Platform::Pose pose;
            image.poseID = platform.poses.size();

            const openMVG::geometry::Pose3 poseMVG(sfm_data.GetPoseOrDie(view.second.get()));
            pose.R = poseMVG.rotation();
            pose.C = poseMVG.center();
            platform.poses.push_back(std::move(pose));
            image.UpdateCamera(platforms);
            nCalibratedImages++;
            images.push_back(std::move(image));
        }
    }

    auto& landmarks = sfm_data.GetLandmarks();
    pointcloud.points.Resize(landmarks.size());
    pointcloud.pointViews.Resize(landmarks.size());
    pointcloud.pointWeights.Release();

    for(int i = 0; i < static_cast<int>(landmarks.size()); i++)
    {
        auto landmarks_iterator = landmarks.cbegin();
        std::advance(landmarks_iterator, i);

        const auto landmark = landmarks_iterator->second;


        auto& point = pointcloud.points[i];
        auto& views = pointcloud.pointViews[i];

        point = landmark.X.cast<float>();
        for (const auto& observation: landmark.obs)
        {
            const auto it = map_view.find(observation.first);
            if (it != map_view.end())
                views.emplace_back(it->second);
        }
        std::sort(views.begin(), views.end(), [](const auto& view0, const auto& view1) { return view0 < view1; });
    }

    for (size_t p=0; p < platforms.size(); ++p)
    {
        MVS::Platform& platform = platforms[p];
        for (size_t c=0; c < platform.cameras.size(); ++c)
        {
            MVS::Platform::Camera& camera = platform.cameras[c];
            // find one image using this camera
            MVS::Image* pImage(nullptr);
            for (MVS::Image& image: images)
            {
                if (image.platformID == p && image.cameraID == c)
                {
                    pImage = &image;
                    break;
                }
            }
            // read image meta-data
            const double fScale(1.0/std::max(pImage->GetSize().width, pImage->GetSize().height));
            camera.K(0, 0) *= fScale;
            camera.K(1, 1) *= fScale;
            camera.K(0, 2) *= fScale;
            camera.K(1, 2) *= fScale;
        }
    }
}

void Scene::Release()
{
    return MVS::Scene::Release();
}

bool Scene::IsEmpty() const
{
    return  MVS::Scene::IsEmpty();
}

bool Scene::Save(const String &fileName, ARCHIVE_TYPE type) const
{
    return MVS::Scene::Save(fileName, type);
}

bool Scene::SelectNeighborViews(uint32_t ID, IndexArr &points, unsigned nMinViews, unsigned nMinPointViews, float fOptimAngle)
{
    return MVS::Scene::SelectNeighborViews(ID, points, nMinViews, nMinPointViews, fOptimAngle);
}

bool Scene::FilterNeighborViews(MVS::ViewScoreArr &neighbors, float fMinArea, float fMinScale, float fMaxScale, float fMinAngle, float fMaxAngle, unsigned nMaxViews)
{
    return MVS::Scene::FilterNeighborViews(neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews);
}

bool Scene::DenseReconstruction(int nFusionMode)
{
    DenseDepthMapData data(*this, nFusionMode);

    // estimate depth-maps
    if (!ComputeDepthMaps(data))
        return false;
    if (ABS(nFusionMode) == 1)
        return true;

    // fuse all depth-maps
    pointcloud.Release();
    data.depthMaps.FuseDepthMaps(pointcloud, MVS::OPTDENSE::nEstimateColors == 2, MVS::OPTDENSE::nEstimateNormals == 2);

    if (!pointcloud.IsEmpty())
    {
        if (pointcloud.colors.IsEmpty() && MVS::OPTDENSE::nEstimateColors == 1)
            EstimatePointColors(images, pointcloud);
        if (pointcloud.normals.IsEmpty() && MVS::OPTDENSE::nEstimateNormals == 1)
            EstimatePointNormals(images, pointcloud);
    }
    return true;
}

Scene::Scene() :
MVS::Scene(0)
{

}

bool Scene::ComputeDepthMaps(DenseDepthMapData &data)
{
    {
        // maps global view indices to our list of views to be processed
        MVS::IIndexArr imagesMap;
        // prepare images for dense reconstruction (load if needed)
        {
            data.images.Reserve(images.GetSize());
            imagesMap.Resize(images.GetSize());
#pragma omp parallel for shared(data)
            for (int_t ID = 0; ID < (int_t)images.GetSize(); ++ID)
            {
                const auto idxImage = (MVS::IIndex)ID;
                // skip invalid, uncalibrated or discarded images
                MVS::Image& imageData = images[idxImage];

                // map image index
#pragma omp critical
                {
                    imagesMap[idxImage] = data.images.GetSize();
                    data.images.Insert(idxImage);
                }
                // reload image at the appropriate resolution
                const unsigned nMaxResolution(imageData.RecomputeMaxResolution(MVS::OPTDENSE::nResolutionLevel, MVS::OPTDENSE::nMinResolution, MVS::OPTDENSE::nMaxResolution));
                imageData.scale = imageData.ResizeImage(nMaxResolution);
                imageData.UpdateCamera(platforms);
            }
        }
        // select images to be used for dense reconstruction
        {
            // for each image, find all useful neighbor views
            MVS::IIndexArr invalidIDs;
#pragma omp parallel for shared(data, invalidIDs)
            for (int_t ID = 0; ID<(int_t)data.images.GetSize(); ++ID)
            {
                const auto idx = (MVS::IIndex)ID;
                const auto idxImage = data.images[idx];
                ASSERT(imagesMap[idxImage] != NO_ID);
                MVS::DepthData& depthData(data.depthMaps.arrDepthData[idxImage]);
                if (!data.depthMaps.SelectViews(depthData))
                {
#pragma omp critical
                    invalidIDs.InsertSort(idx);
                }
            }
            RFOREACH(i, invalidIDs)
            {
                const auto idx = invalidIDs[i];
                imagesMap[data.images.Last()] = idx;
                imagesMap[data.images[idx]] = NO_ID;
                data.images.RemoveAt(idx);
            }
            // globally select a target view for each reference image
            if (MVS::OPTDENSE::nNumViews == 1 && !data.depthMaps.SelectViews(data.images, imagesMap, data.neighborsMap))
            {
                VERBOSE("error: no valid images to be dense reconstructed");
                return false;
            }
            ASSERT(!data.images.IsEmpty());
        }
    }

    // initialize the queue of images to be processed
    data.idxImage = 0;
    ASSERT(data.events.IsEmpty());
    // start working threads
    data.events.AddEvent(new EVTProcessImage(0));
    data.progress = new Util::Progress("Estimated depth-maps", data.images.GetSize());

    //if (nMaxThreads > 1)
    //{
    // multi-thread execution
    //    cList<SEACAVE::Thread> threads(2);
    //FOREACHPTR(pThread, threads)
    //       pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
    //FOREACHPTR(pThread, threads)
    //        pThread->join();
    //DenseReconstructionEstimate(this);
    //} else
    //{
    // single-thread execution

    DenseReconstructionEstimate(data);
    //}

    if (!data.events.IsEmpty())
    {
        data.events.Clear();
    }
    data.progress.Release();

    if ((MVS::OPTDENSE::nOptimize & MVS::OPTDENSE::ADJUST_FILTER) != 0)
    {
        // initialize the queue of depth-maps to be filtered
        data.sem.Clear();
        data.idxImage = data.images.GetSize();
        ASSERT(data.events.IsEmpty());
        FOREACH(i, data.images)
                data.events.AddEvent(new EVTFilterDepthMap(i));
        // start working threads
        data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
        //if (nMaxThreads > 1)
        //{
        //    // multi-thread execution
        //    cList<SEACAVE::Thread> threads(MINF(nMaxThreads, (unsigned)data.images.GetSize()));
        //    FOREACHPTR(pThread, threads)
        //            pThread->start(DenseReconstructionFilterTmp, (void*)&data);
        //    FOREACHPTR(pThread, threads)
        //            pThread->join();
        //} else
        //{
        // single-thread execution
        DenseReconstructionFilter(data);
        //}
        if (!data.events.IsEmpty())
        {
            data.events.Clear();
        }
        data.progress.Release();
    }
    return true;
}

void Scene::DenseReconstructionEstimate(DenseDepthMapData& data)
{
    while (true)
    {
        auto evt = data.events.GetEvent();
        switch (evt->GetID())
        {
        case EVT_PROCESSIMAGE:
        {
            const EVTProcessImage& evtImage = *((EVTProcessImage*)evt);
            if (evtImage.idxImage >= data.images.GetSize())
            {
                if (nMaxThreads > 1)
                {
                    // close working threads
                    data.events.AddEvent(new EVTClose);
                }
                return;
            }
            // select views to reconstruct the depth-map for this image
            const auto idx = data.images[evtImage.idxImage];
            auto& depthData(data.depthMaps.arrDepthData[idx]);
            // init images pair: reference image and the best neighbor view
            ASSERT(data.neighborsMap.IsEmpty() || data.neighborsMap[evtImage.idxImage] != NO_ID);
            if (!data.depthMaps.InitViews(depthData, data.neighborsMap.IsEmpty() ? NO_ID : data.neighborsMap[evtImage.idxImage], MVS::OPTDENSE::nNumViews))
            {
                // process next image
                data.events.AddEvent(new EVTProcessImage((MVS::IIndex)Thread::safeInc(data.idxImage)));
                break;
            }

            // estimate depth-map
            data.events.AddEventFirst(new EVTEstimateDepthMap(evtImage.idxImage));
            break;
        }

        case EVT_ESTIMATEDEPTHMAP:
        {
            const EVTEstimateDepthMap& evtImage = *((EVTEstimateDepthMap*)evt);
            // request next image initialization to be performed while computing this depth-map
            data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
            // extract depth map
            data.sem.Wait();
            if (data.nFusionMode >= 0)
            {
                // extract depth-map using Patch-Match algorithm
                data.depthMaps.EstimateDepthMap(data.images[evtImage.idxImage]);
            } else
            {
                // extract disparity-maps using SGM algorithm
                if (data.nFusionMode == -1)
                {
                    data.sgm.Match(*this, data.images[evtImage.idxImage], MVS::OPTDENSE::nNumViews);
                } else
                {
                    // fuse existing disparity-maps
                    const auto idx(data.images[evtImage.idxImage]);
                    auto& depthData(data.depthMaps.arrDepthData[idx]);
                    data.sgm.Fuse(*this, data.images[evtImage.idxImage], MVS::OPTDENSE::nNumViews, 2, depthData.depthMap, depthData.confMap);
                    if (MVS::OPTDENSE::nEstimateNormals == 2)
                    {
                        MVS::EstimateNormalMap(depthData.images.front().camera.K, depthData.depthMap, depthData.normalMap);
                    }
                    depthData.dMin = ZEROTOLERANCE<float>(); depthData.dMax = FLT_MAX;

                }
            }
            data.sem.Signal();
            if (MVS::OPTDENSE::nOptimize & MVS::OPTDENSE::OPTIMIZE)
            {
                // optimize depth-map
                data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
            } else
            {
                // save depth-map
                data.progress->operator++();
            }
            break;
        }

        case EVT_OPTIMIZEDEPTHMAP:
        {
            const EVTOptimizeDepthMap& evtImage = *((EVTOptimizeDepthMap*)evt);
            const auto idx = data.images[evtImage.idxImage];
            auto& depthData(data.depthMaps.arrDepthData[idx]);
            // apply filters
            if (MVS::OPTDENSE::nOptimize & (MVS::OPTDENSE::REMOVE_SPECKLES))
            {
                TD_TIMER_START();
                if (data.depthMaps.RemoveSmallSegments(depthData))
                {
                    DEBUG_ULTIMATE("Depth-map %3u filtered: remove small segments (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
                }
            }
            if (MVS::OPTDENSE::nOptimize & (MVS::OPTDENSE::FILL_GAPS))
            {
                if (data.depthMaps.GapInterpolation(depthData))
                {

                }
            }
            // save depth-map
            data.progress->operator++();
            break;
        }

        case EVT_CLOSE:
        {
            return;
        }

        default:
            ASSERT("Should not happen!" == NULL);
        }
        delete evt;
    }
}

void Scene::DenseReconstructionFilter(DenseDepthMapData& data)
{
    Event* evt;
    while ((evt=data.events.GetEvent(0)) != NULL)
    {
        switch (evt->GetID())
        {
        case EVT_FILTERDEPTHMAP:
        {
            const EVTFilterDepthMap& evtImage = *((EVTFilterDepthMap*)evt);
            const auto idx = data.images[evtImage.idxImage];
            auto& depthData(data.depthMaps.arrDepthData[idx]);
            if (!depthData.IsValid())
            {
                data.SignalCompleteDepthmapFilter();
                break;
            }
            // make sure all depth-maps are loaded

            const unsigned numMaxNeighbors(8);
            MVS::IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
            FOREACH(n, depthData.neighbors)
            {
                const auto idxView = depthData.neighbors[n].idx.ID;
                auto& depthDataPair = data.depthMaps.arrDepthData[idxView];
                if (!depthDataPair.IsValid())
                    continue;
                idxNeighbors.Insert(n);
                if (idxNeighbors.GetSize() == numMaxNeighbors)
                    break;
            }
            // filter the depth-map for this image
            if (data.depthMaps.FilterDepthMap(depthData, idxNeighbors, MVS::OPTDENSE::bFilterAdjust))
            {
                // load the filtered maps after all depth-maps were filtered
                data.progress->operator++();
            }
            // unload referenced depth-maps

            data.SignalCompleteDepthmapFilter();
            break;
        }

        case EVT_FAIL:
        {
            data.events.AddEventFirst(new EVTFail);
            delete evt;
            return;
        }

        default:
            ASSERT("Should not happen!" == NULL);
        }
        delete evt;
    }
}

bool Scene::RefineMeshCUDA(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize, unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fGradientStep)
{
    return MVS::Scene::RefineMeshCUDA(nResolutionLevel, nMinResolution, nMaxViews, fDecimateMesh, nCloseHoles, nEnsureEdgeSize, nMaxFaceArea, nScales, fScaleStep, nAlternatePair, fRegularityWeight, fRatioRigidityElasticity, fGradientStep);
}

bool Scene::ReconstructMesh(float distInsert, bool bUseFreeSpaceSupport, unsigned nItersFixNonManifold, float kSigma, float kQual, float kb, float kf, float kRel, float kAbs, float kOutl, float kInf)
{
    return MVS::Scene::ReconstructMesh(distInsert, bUseFreeSpaceSupport, nItersFixNonManifold, kSigma, kQual, kb, kf, kRel, kAbs, kOutl, kInf);
}

bool Scene::TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, float fOutlierThreshold, float fRatioDataSmoothness, bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty)
{
    return MVS::Scene::TextureMesh(nResolutionLevel, nMinResolution, fOutlierThreshold, fRatioDataSmoothness, bGlobalSeamLeveling, bLocalSeamLeveling, nTextureSizeMultiple, nRectPackingHeuristic, colEmpty);
}


