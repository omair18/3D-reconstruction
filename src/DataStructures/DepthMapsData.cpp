#include "Scene.h"
#include "DepthMapsData.h"
#include "DepthEstimator.h"

#include <OpenMVS/MVS/SceneDensify.h>
#include <OpenMVS/Math/TRWS/typeGeneral.h>
#include <OpenMVS/Math/TRWS/typeBinary.h>
#include <OpenMVS/Math/TRWS/MRFEnergy.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

DepthMapsData::DepthMapsData(Scene &_scene)
    :scene(_scene),
      arrDepthData(_scene.images.GetSize())
{

}

DepthMapsData::~DepthMapsData()
{

}

void* EstimateDepthMapTmp(void* arg)
{
    DepthEstimator& estimator = *((DepthEstimator*)arg);
    IDX idx;
    std::chrono::high_resolution_clock clk;
    auto start = clk.now();
    while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
        estimator.ProcessPixel(idx);
    auto end = clk.now();
    //std::cout << "TIME : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    return NULL;
}

void* ScoreDepthMapTmp(void* arg)
{
    DepthEstimator& estimator = *((DepthEstimator*)arg);
    IDX idx;
    while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
    {
        const ImageRef& x = estimator.coords[idx];
        if (!estimator.PreparePixelPatch(x) || !estimator.FillPixelPatch())
        {
            estimator.depthMap0(x) = 0;
            estimator.normalMap0(x) = MVS::Normal::ZERO;
            estimator.confMap0(x) = 2.f;
            continue;
        }
        auto& depth = estimator.depthMap0(x);
        auto& normal = estimator.normalMap0(x);
        const MVS::Normal viewDir(Cast<float>(static_cast<const Point3&>(estimator.X0)));
        if (!ISINSIDE(depth, estimator.dMin, estimator.dMax))
        {
            // init with random values
            depth = estimator.RandomDepth(estimator.dMinSqr, estimator.dMaxSqr);
            normal = estimator.RandomNormal(viewDir);
        } else if (normal.dot(viewDir) >= 0)
        {
            // replace invalid normal with random values
            normal = estimator.RandomNormal(viewDir);
        }
        estimator.confMap0(x) = estimator.ScorePixel(depth, normal);
    }
    return NULL;
}

void* STCALL EndDepthMapTmp(void* arg)
{
    DepthEstimator& estimator = *((DepthEstimator*)arg);
    IDX idx;
    const float fOptimAngle(FD2R(MVS::OPTDENSE::fOptimAngle));
    while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
    {
        const ImageRef& x = estimator.coords[idx];
        ASSERT(estimator.depthMap0(x) >= 0);
        auto& depth = estimator.depthMap0(x);
        float& conf = estimator.confMap0(x);
        // check if the score is good enough
        // and that the cross-estimates is close enough to the current estimate
        if (depth <= 0 || conf >= MVS::OPTDENSE::fNCCThresholdKeep)
        {
            conf = 0;
            estimator.normalMap0(x) = MVS::Normal::ZERO;
            depth = 0;
        } else
        {
            // converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
            conf = conf>=1.f ? 0.f : 1.f-conf;
        }
    }
    return NULL;
}

bool DepthMapsData::SelectViews(MVS::IIndexArr &images, MVS::IIndexArr &imagesMap, MVS::IIndexArr &neighborsMap)
{
    // find all pair of images valid for dense reconstruction
    std::unordered_map<uint64_t,float> edges;
    double totScore = 0;
    unsigned numScores = 0;
    FOREACH(i, images)
    {
        const uint32_t idx = images[i];
        ASSERT(imagesMap[idx] != NO_ID);
        const auto& neighbors(arrDepthData[idx].neighbors);
        ASSERT(neighbors.GetSize() <= MVS::OPTDENSE::nMaxViews);
        // register edges
        FOREACHPTR(pNeighbor, neighbors)
        {
            const auto idx2(pNeighbor->idx.ID);
            ASSERT(imagesMap[idx2] != NO_ID);
            edges[MakePairIdx(idx,idx2)] = pNeighbor->idx.area;
            totScore += pNeighbor->score;
            ++numScores;
        }
    }
    if (edges.empty())
        return false;
    const float avgScore((float)(totScore/(double)numScores));

    // run global optimization
    const float fPairwiseMul = MVS::OPTDENSE::fPairwiseMul; // default 0.3
    const float fEmptyUnaryMult = 6.f;
    const float fEmptyPairwise = 8.f * MVS::OPTDENSE::fPairwiseMul;
    const float fSamePairwise = 24.f * MVS::OPTDENSE::fPairwiseMul;
    const uint32_t _num_labels = MVS::OPTDENSE::nMaxViews+1; // N neighbors and an empty state
    const uint32_t _num_nodes = images.GetSize();
    CAutoPtr<MRFEnergy<TypeGeneral>> energy(new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
    CAutoPtrArr<MRFEnergy<TypeGeneral>::NodeId> nodes(new MRFEnergy<TypeGeneral>::NodeId[_num_nodes]);
    typedef SEACAVE::cList<TypeGeneral::REAL, const TypeGeneral::REAL&, 0> EnergyCostArr;
    // unary costs: inverse proportional to the image pair score
    EnergyCostArr arrUnary(_num_labels);
    for (uint32_t n=0; n<_num_nodes; ++n)
    {
        const auto& neighbors(arrDepthData[images[n]].neighbors);
        FOREACH(k, neighbors)
                arrUnary[k] = avgScore/neighbors[k].score; // use average score to normalize the values (not to depend so much on the number of features in the scene)
        arrUnary[neighbors.GetSize()] = fEmptyUnaryMult*(neighbors.IsEmpty()?avgScore*0.01f:arrUnary[neighbors.GetSize()-1]);
        nodes[n] = energy->AddNode(TypeGeneral::LocalSize(neighbors.GetSize()+1), TypeGeneral::NodeData(arrUnary.Begin()));
    }
    // pairwise costs: as ratios between the area to be covered and the area actually covered
    EnergyCostArr arrPairwise(_num_labels*_num_labels);
    for (const auto& edge: edges)
    {
        const PairIdx pair(edge.first);
        const float area(edge.second);
        const auto& neighborsI(arrDepthData[pair.i].neighbors);
        const auto& neighborsJ(arrDepthData[pair.j].neighbors);
        arrPairwise.Empty();
        FOREACHPTR(pNj, neighborsJ)
        {
            const uint32_t i(pNj->idx.ID);
            const float areaJ(area/pNj->idx.area);
            FOREACHPTR(pNi, neighborsI)
            {
                const uint32_t j(pNi->idx.ID);
                const float areaI(area/pNi->idx.area);
                arrPairwise.Insert(pair.i == i && pair.j == j ? fSamePairwise : fPairwiseMul*(areaI+areaJ));
            }
            arrPairwise.Insert(fEmptyPairwise+fPairwiseMul*areaJ);
        }
        for (const auto& Ni: neighborsI)
        {
            const float areaI(area/Ni.idx.area);
            arrPairwise.Insert(fPairwiseMul*areaI+fEmptyPairwise);
        }
        arrPairwise.Insert(fEmptyPairwise*2);
        const uint32_t nodeI(imagesMap[pair.i]);
        const uint32_t nodeJ(imagesMap[pair.j]);
        energy->AddEdge(nodes[nodeI], nodes[nodeJ], TypeGeneral::EdgeData(TypeGeneral::GENERAL, arrPairwise.Begin()));
    }

    // minimize energy
    MRFEnergy<TypeGeneral>::Options options;
    options.m_eps = MVS::OPTDENSE::fOptimizerEps;
    options.m_iterMax = MVS::OPTDENSE::nOptimizerMaxIters;

    //options.m_printIter = 1;
    //options.m_printMinIter = 1;

    TypeGeneral::REAL energyVal, lowerBound;
    energy->Minimize_TRW_S(options, lowerBound, energyVal);

    // extract optimized depth map
    neighborsMap.Resize(_num_nodes);
    for (uint32_t n=0; n<_num_nodes; ++n)
    {
        const auto& neighbors(arrDepthData[images[n]].neighbors);
        uint32_t& idxNeighbor = neighborsMap[n];
        const uint32_t label((uint32_t)energy->GetSolution(nodes[n]));
        ASSERT(label <= neighbors.GetSize());
        if (label == neighbors.GetSize())
        {
            idxNeighbor = NO_ID; // empty
        } else
        {
            idxNeighbor = label;
            DEBUG_ULTIMATE("\treference image %3u paired with target image %3u (idx %2u)", images[n], neighbors[label].idx.ID, label);
        }
    }

    // remove all images with no valid neighbors
    RFOREACH(i, neighborsMap)
    {
        if (neighborsMap[i] == NO_ID)
        {
            // remove image with no neighbors
            for (auto& imageMap: imagesMap)
                if (imageMap != NO_ID && imageMap > i)
                    --imageMap;
            imagesMap[images[i]] = NO_ID;
            images.RemoveAtMove(i);
            neighborsMap.RemoveAtMove(i);
        }
    }
    return !images.IsEmpty();
}

bool DepthMapsData::SelectViews(MVS::DepthData &depthData)
{
    // find and sort valid neighbor views
    const auto idxImage = (uint32_t)(&depthData-arrDepthData.Begin());
    ASSERT(depthData.neighbors.IsEmpty());
    ASSERT(scene.images[idxImage].neighbors.IsEmpty());
    if (!scene.SelectNeighborViews(idxImage, depthData.points, MVS::OPTDENSE::nMinViews, MVS::OPTDENSE::nMinViewsTrustPoint > 1 ? MVS::OPTDENSE::nMinViewsTrustPoint : 2,
                                   FD2R(MVS::OPTDENSE::fOptimAngle)))
        return false;
    depthData.neighbors.CopyOf(scene.images[idxImage].neighbors);

    // remove invalid neighbor views
    const float fMinAngle(FD2R(MVS::OPTDENSE::fMinAngle));
    const float fMaxAngle(FD2R(MVS::OPTDENSE::fMaxAngle));
    if (!Scene::FilterNeighborViews(depthData.neighbors, MVS::OPTDENSE::fMinArea, 0.2, 3.2, fMinAngle, fMaxAngle, MVS::OPTDENSE::nMaxViews))
    {
        DEBUG_EXTRA("error: reference image %3u has no good images in view", idxImage);
        return false;
    }
    return true;
}

bool DepthMapsData::InitViews(MVS::DepthData &depthData, uint32_t idxNeighbor, uint32_t numNeighbors)
{
    const auto idxImage = ((uint32_t)(&depthData-arrDepthData.Begin()));
    ASSERT(!depthData.neighbors.IsEmpty());
    ASSERT(depthData.images.IsEmpty());

    // set this image the first image in the array
    depthData.images.Reserve(depthData.neighbors.GetSize()+1);
    depthData.images.AddEmpty();

    if (idxNeighbor != NO_ID)
    {
        // set target image as the given neighbor
        const auto& neighbor = depthData.neighbors[idxNeighbor];
        auto& imageTrg = depthData.images.AddEmpty();
        imageTrg.pImageData = &scene.images[neighbor.idx.ID];
        imageTrg.scale = neighbor.idx.scale;
        imageTrg.camera = imageTrg.pImageData->camera;
        imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
        if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
            imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
        DEBUG_EXTRA("Reference image %3u paired with image %3u", idxImage, neighbor.idx.ID);
    } else
    {
        // init all neighbor views too (global reconstruction is used)
        const float fMinScore(MAXF(depthData.neighbors.First().score * (MVS::OPTDENSE::fViewMinScoreRatio * 0.1f), MVS::OPTDENSE::fViewMinScore));
        FOREACH(idx, depthData.neighbors)
        {
            const auto& neighbor = depthData.neighbors[idx];
            if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
                    (neighbor.score < fMinScore))
                break;
            auto& imageTrg = depthData.images.AddEmpty();
            imageTrg.pImageData = &scene.images[neighbor.idx.ID];
            imageTrg.scale = neighbor.idx.scale;
            imageTrg.camera = imageTrg.pImageData->camera;
            imageTrg.pImageData->image.toGray(imageTrg.image, cv::COLOR_BGR2GRAY, true);
            if (imageTrg.ScaleImage(imageTrg.image, imageTrg.image, imageTrg.scale))
                imageTrg.camera = imageTrg.pImageData->GetCamera(scene.platforms, imageTrg.image.size());
        }
    }
    if (depthData.images.GetSize() < 2)
    {
        depthData.images.Release();
        return false;
    }

    // init the first image as well
    auto& imageRef = depthData.images.First();
    imageRef.scale = 1;
    imageRef.pImageData = &scene.images[idxImage];
    imageRef.pImageData->image.toGray(imageRef.image, cv::COLOR_BGR2GRAY, true);
    imageRef.camera = imageRef.pImageData->camera;
    return true;
}

bool DepthMapsData::InitDepthMap(MVS::DepthData &depthData)
{
    TD_TIMER_STARTD();

    ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
    const auto& image(depthData.GetView());
    TriangulatePoints2DepthMap(image, scene.pointcloud, depthData.points, depthData.depthMap, depthData.normalMap, depthData.dMin, depthData.dMax);
    depthData.dMin *= 0.9f;
    depthData.dMax *= 1.1f;
    DEBUG_ULTIMATE("Depth-map %3u roughly estimated from %u sparse points: %dx%d (%s)", image.GetID(), depthData.points.size(), image.image.width(), image.image.height(), TD_TIMER_GET_FMT().c_str());
    return true;
}

bool DepthMapsData::EstimateDepthMap(uint32_t idxImage)
{
    TD_TIMER_STARTD();
    // initialize depth and normal maps
    auto& depthData(arrDepthData[idxImage]);
    ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());

    const auto& image(depthData.images.First());
    ASSERT(!image.image.empty() && !depthData.images[1].image.empty());

    const Image8U::Size size(image.image.size());
    depthData.depthMap.create(size); depthData.depthMap.memset(0);
    depthData.normalMap.create(size);
    depthData.confMap.create(size);
    const unsigned nMaxThreads(scene.nMaxThreads);

    // initialize the depth-map
    if (MVS::OPTDENSE::nMinViewsTrustPoint < 2)
    {
        // compute depth range and initialize known depths
        const int nPixelArea(2); // half windows size around a pixel to be initialize with the known depth
        const auto& camera = depthData.images.First().camera;
        depthData.dMin = FLT_MAX;
        depthData.dMax = 0;
        for(auto& point: depthData.points)
        {
            const auto& X = scene.pointcloud.points[point];
            const Point3 camX(camera.TransformPointW2C(Cast<REAL>(X)));
            const ImageRef x(ROUND2INT(camera.TransformPointC2I(camX)));
            const float d((float)camX.z);
            const ImageRef sx(MAXF(x.x-nPixelArea,0), MAXF(x.y-nPixelArea,0));
            const ImageRef ex(MINF(x.x+nPixelArea,size.width-1), MINF(x.y+nPixelArea,size.height-1));
            for (int y=sx.y; y <= ex.y; ++y)
            {
                for (int x = sx.x; x <= ex.x; ++x)
                {
                    depthData.depthMap(y,x) = d;
                    depthData.normalMap(y,x) = MVS::Normal::ZERO;
                }
            }
            if (depthData.dMin > d)
                depthData.dMin = d;
            if (depthData.dMax < d)
                depthData.dMax = d;
        }
        depthData.dMin *= 0.9f;
        depthData.dMax *= 1.1f;
    } else
    {
        // compute rough estimates using the sparse point-cloud
        InitDepthMap(depthData);
    }

    // init integral images and index to image-ref map for the reference data
    MVS::DepthEstimator::WeightMap weightMap0(size.area()-(size.width + 1) * MVS::DepthEstimator::nSizeHalfWindow);

    if (prevDepthMapSize != size)
    {
        prevDepthMapSize = size;
        BitMatrix mask;
        MVS::DepthEstimator::MapMatrix2ZigzagIdx(size, coords, mask, MAXF(64,(int)nMaxThreads*8));
    }

    // init threads
    //ASSERT(nMaxThreads > 0);
    cList<MVS::DepthEstimator> estimators;
    estimators.Reserve(nMaxThreads);
    cList<SEACAVE::Thread> threads;
    if (nMaxThreads > 1)
        threads.Resize(nMaxThreads-1); // current thread is also used
    volatile Thread::safe_t idxPixel;

    // initialize the reference confidence map (NCC score map) with the score of the current estimates
    {
        // create working threads
        idxPixel = -1;
        DepthEstimator estimator(0, depthData, idxPixel, weightMap0, coords);
        //ASSERT(estimators.IsEmpty());
        while (estimators.GetSize() < nMaxThreads)
            estimators.AddConstruct(0, depthData, idxPixel, weightMap0, coords);
        //ASSERT(estimators.GetSize() == threads.GetSize()+1);
        FOREACH(i, threads)
                threads[i].start(ScoreDepthMapTmp, &estimators[i]);
        ScoreDepthMapTmp(&estimator);
        // wait for the working threads to close
        FOREACHPTR(pThread, threads)
               pThread->join();
        estimators.Release();
    }

    // run propagation and random refinement cycles on the reference data
    for (unsigned iter=0; iter< MVS::OPTDENSE::nEstimationIters; ++iter)
    {
        // create working threads
        idxPixel = -1;
        MVS::DepthEstimator estimator(iter, depthData, idxPixel,
                                      weightMap0,
                                      coords);
        //ASSERT(estimators.IsEmpty());
        while (estimators.GetSize() < nMaxThreads)
            estimators.AddConstruct(iter, depthData, idxPixel,
                                    weightMap0,
                                    coords);
        //ASSERT(estimators.GetSize() == threads.GetSize()+1);
        FOREACH(i, threads)
                threads[i].start(EstimateDepthMapTmp, &estimators[i]);
        EstimateDepthMapTmp(&estimator);
        // wait for the working threads to close
        FOREACHPTR(pThread, threads)
                pThread->join();
        estimators.Release();
   }

    // remove all estimates with too big score and invert confidence map
    {
        // create working threads
        idxPixel = -1;
        MVS::DepthEstimator estimator(0, depthData, idxPixel, weightMap0, coords);
        //ASSERT(estimators.IsEmpty());
        while (estimators.GetSize() < nMaxThreads)
            estimators.AddConstruct(0, depthData, idxPixel, weightMap0, coords);
        //ASSERT(estimators.GetSize() == threads.GetSize()+1);
        FOREACH(i, threads)
                threads[i].start(EndDepthMapTmp, &estimators[i]);
        EndDepthMapTmp(&estimator);
        // wait for the working threads to close
        FOREACHPTR(pThread, threads)
                pThread->join();
        estimators.Release();
    }

    DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", image.GetID(),
                depthData.images.GetSize() > 2 ?
                    String::FormatString("estimated using %2u images", depthData.images.GetSize()-1).c_str() :
                    String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
            size.width, size.height, TD_TIMER_GET_FMT().c_str());
    return true;
}

bool DepthMapsData::RemoveSmallSegments(MVS::DepthData &depthData)
{
    const float fDepthDiffThreshold(MVS::OPTDENSE::fDepthDiffThreshold*0.7f);
    unsigned speckle_size = MVS::OPTDENSE::nSpeckleSize;
    auto& depthMap = depthData.depthMap;
    auto& normalMap = depthData.normalMap;
    auto& confMap = depthData.confMap;
    ASSERT(!depthMap.empty());
    const ImageRef size(depthMap.size());

    // allocate memory on heap for dynamic programming arrays
    TImage<bool> done_map(size, false);
    ImageRef seg_list[size.x*size.y];
    unsigned seg_list_count;
    unsigned seg_list_curr;
    ImageRef neighbor[4];

    // for all pixels do
    for (int u=0; u<size.x; ++u)
    {
        for (int v=0; v<size.y; ++v)
        {
            // if the first pixel in this segment has been already processed => skip
            if (done_map(v,u))
                continue;

            // init segment list (add first element
            // and set it to be the next element to check)
            seg_list[0] = ImageRef(u,v);
            seg_list_count = 1;
            seg_list_curr  = 0;

            // add neighboring segments as long as there
            // are none-processed pixels in the seg_list;
            // none-processed means: seg_list_curr<seg_list_count
            while (seg_list_curr < seg_list_count)
            {
                // get address of current pixel in this segment
                const ImageRef addr_curr(seg_list[seg_list_curr]);
                const auto& depth_curr = depthMap(addr_curr);

                if (depth_curr > 0)
                {
                    // fill list with neighbor positions
                    neighbor[0] = ImageRef(addr_curr.x-1, addr_curr.y  );
                    neighbor[1] = ImageRef(addr_curr.x+1, addr_curr.y  );
                    neighbor[2] = ImageRef(addr_curr.x  , addr_curr.y-1);
                    neighbor[3] = ImageRef(addr_curr.x  , addr_curr.y+1);

                    // for all neighbors do
                    for (int i=0; i<4; ++i)
                    {
                        // get neighbor pixel address
                        const ImageRef& addr_neighbor(neighbor[i]);
                        // check if neighbor is inside image
                        if (addr_neighbor.x>=0 && addr_neighbor.y>=0 && addr_neighbor.x<size.x && addr_neighbor.y<size.y)
                        {
                            // check if neighbor has not been added yet
                            bool& done = done_map(addr_neighbor);
                            if (!done)
                            {
                                // check if the neighbor is valid and similar to the current pixel
                                // (belonging to the current segment)
                                const auto& depth_neighbor = depthMap(addr_neighbor);
                                if (depth_neighbor>0 && IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold))
                                {
                                    // add neighbor coordinates to segment list
                                    seg_list[seg_list_count++] = addr_neighbor;
                                    // set neighbor pixel in done_map to "done"
                                    // (otherwise a pixel may be added 2 times to the list, as
                                    //  neighbor of one pixel and as neighbor of another pixel)
                                    done = true;
                                }
                            }
                        }
                    }
                }

                // set current pixel in seg_list to "done"
                ++seg_list_curr;

                // set current pixel in done_map to "done"
                done_map(addr_curr) = true;
            } // end: while (seg_list_curr < seg_list_count)

            // if segment NOT large enough => invalidate pixels
            if (seg_list_count < speckle_size)
            {
                // for all pixels in current segment invalidate pixels
                for (unsigned i=0; i<seg_list_count; ++i)
                {
                    depthMap(seg_list[i]) = 0;
                    if (!normalMap.empty()) normalMap(seg_list[i]) = MVS::Normal::ZERO;
                    if (!confMap.empty()) confMap(seg_list[i]) = 0;
                }
            }
        }
    }
    return true;
}

bool DepthMapsData::GapInterpolation(MVS::DepthData &depthData)
{
    const float fDepthDiffThreshold = MVS::OPTDENSE::fDepthDiffThreshold * 2.5f;
    auto& depthMap = depthData.depthMap;
    auto& normalMap = depthData.normalMap;
    auto& confMap = depthData.confMap;
    ASSERT(!depthMap.empty());
    const ImageRef size(depthMap.size());

    // 1. Row-wise:
    // for each row do
    for (int v = 0; v < size.y; ++v)
    {
        // init counter
        unsigned count = 0;

        // for each element of the row do
        for (int u = 0; u < size.x; ++u)
        {
            // get depth of this location
            const auto& depth = depthMap(v,u);

            // if depth not valid => count and skip it
            if (depth <= 0)
            {
                ++count;
                continue;
            }
            if (count == 0)
                continue;

            // check if speckle is small enough
            // and value in range
            if (count <= MVS::OPTDENSE::nIpolGapSize && (unsigned)u > count)
            {
                // first value index for interpolation
                int u_curr(u-count);
                const int u_first(u_curr-1);
                // compute mean depth
                const auto& depthFirst = depthMap(v,u_first);
                if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold))
                {
                    // interpolate values
                    const auto diff = (depth - depthFirst) / (count + 1);
                    auto d = depthFirst;
                    const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
                    if (normalMap.empty())
                    {
                        do
                        {
                            depthMap(v,u_curr) = (d+=diff);
                            if (!confMap.empty()) confMap(v,u_curr) = c;
                        } while (++u_curr<u);
                    } else
                    {
                        Point2f dir1, dir2;
                        Normal2Dir(normalMap(v,u_first), dir1);
                        Normal2Dir(normalMap(v,u), dir2);
                        const Point2f dirDiff((dir2-dir1)/float(count+1));
                        do
                        {
                            depthMap(v,u_curr) = (d+=diff);
                            dir1 += dirDiff;
                            Dir2Normal(dir1, normalMap(v,u_curr));
                            if (!confMap.empty()) confMap(v,u_curr) = c;
                        } while (++u_curr<u);
                    }
                }
            }

            // reset counter
            count = 0;
        }
    }

    // 2. Column-wise:
    // for each column do
    for (int u = 0; u < size.x; ++u)
    {

        // init counter
        unsigned count = 0;

        // for each element of the column do
        for (int v=0; v<size.y; ++v)
        {
            // get depth of this location
            const auto& depth = depthMap(v,u);

            // if depth not valid => count and skip it
            if (depth <= 0)
            {
                ++count;
                continue;
            }
            if (count == 0)
                continue;

            // check if gap is small enough
            // and value in range
            if (count <= MVS::OPTDENSE::nIpolGapSize && (unsigned)v > count)
            {
                // first value index for interpolation
                int v_curr(v-count);
                const int v_first(v_curr-1);
                // compute mean depth
                const auto& depthFirst = depthMap(v_first,u);
                if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold))
                {
                    // interpolate values
                    const auto diff = (depth - depthFirst) / (count + 1);
                    auto d = depthFirst;
                    const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
                    if (normalMap.empty())
                    {
                        do
                        {
                            depthMap(v_curr,u) = (d+=diff);
                            if (!confMap.empty()) confMap(v_curr,u) = c;
                        } while (++v_curr<v);
                    } else
                    {
                        Point2f dir1, dir2;
                        Normal2Dir(normalMap(v_first,u), dir1);
                        Normal2Dir(normalMap(v,u), dir2);
                        const Point2f dirDiff((dir2-dir1) / float(count+1));
                        do
                        {
                            depthMap(v_curr,u) = (d+=diff);
                            dir1 += dirDiff;
                            Dir2Normal(dir1, normalMap(v_curr,u));
                            if (!confMap.empty()) confMap(v_curr,u) = c;
                        } while (++v_curr<v);
                    }
                }
            }

            // reset counter
            count = 0;
        }
    }
    return true;
}

bool DepthMapsData::FilterDepthMap(MVS::DepthData &depthDataRef, const MVS::IIndexArr &idxNeighbors, bool bAdjust)
{    
    // count valid neighbor depth-maps
    ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty());
    const auto N = idxNeighbors.GetSize();
    ASSERT(MVS::OPTDENSE::nMinViewsFilter > 0 && scene.nCalibratedImages > 1);
    const auto nMinViews(MINF(MVS::OPTDENSE::nMinViewsFilter,scene.nCalibratedImages-1));
    const auto nMinViewsAdjust(MINF(MVS::OPTDENSE::nMinViewsFilterAdjust,scene.nCalibratedImages-1));
    if (N < nMinViews || N < nMinViewsAdjust)
    {
        DEBUG("error: depth map %3u can not be filtered", depthDataRef.GetView().GetID());
        return false;
    }

    // project all neighbor depth-maps to this image
    const auto& imageRef = depthDataRef.images.First();
    const Image8U::Size sizeRef(depthDataRef.depthMap.size());
    const auto& cameraRef = imageRef.camera;
    MVS::DepthMapArr depthMaps(N);
    MVS::ConfidenceMapArr confMaps(N);
    FOREACH(n, depthMaps)
    {
        auto& depthMap = depthMaps[n];
        depthMap.create(sizeRef);
        depthMap.memset(0);
        auto& confMap = confMaps[n];
        if (bAdjust)
        {
            confMap.create(sizeRef);
            confMap.memset(0);
        }
        const auto idxView = depthDataRef.neighbors[idxNeighbors[(uint32_t)n]].idx.ID;
        const auto& depthData = arrDepthData[idxView];
        const auto& camera = depthData.images.First().camera;
        const Image8U::Size size(depthData.depthMap.size());
        for (int i=0; i<size.height; ++i)
        {
            for (int j=0; j<size.width; ++j)
            {
                const ImageRef x(j,i);
                const MVS::Depth depth(depthData.depthMap(x));
                if (depth == 0)
                    continue;
                ASSERT(depth > 0);
                const Point3 X(camera.TransformPointI2W(Point3(x.x,x.y,depth)));
                const Point3 camX(cameraRef.TransformPointW2C(X));
                if (camX.z <= 0)
                    continue;

                // set depth on the 4 pixels around the image projection
                const Point2 imgX(cameraRef.TransformPointC2I(camX));
                const ImageRef xRefs[4] = {
                    ImageRef(FLOOR2INT(imgX.x), FLOOR2INT(imgX.y)),
                    ImageRef(FLOOR2INT(imgX.x), CEIL2INT(imgX.y)),
                    ImageRef(CEIL2INT(imgX.x), FLOOR2INT(imgX.y)),
                    ImageRef(CEIL2INT(imgX.x), CEIL2INT(imgX.y))
                };
                for (const auto & xRef : xRefs)
                {
                    if (!depthMap.isInside(xRef))
                        continue;
                    auto& depthRef = depthMap(xRef);
                    if (depthRef != 0 && depthRef < (MVS::Depth)camX.z)
                        continue;
                    depthRef = (MVS::Depth)camX.z;
                    if (bAdjust)
                        confMap(xRef) = depthData.confMap(x);
                }

            }
        }
    }

    const float thDepthDiff(MVS::OPTDENSE::fDepthDiffThreshold * 1.2f);
    MVS::DepthMap newDepthMap(sizeRef);
    MVS::ConfidenceMap newConfMap(sizeRef);
    if (bAdjust)
    {
        // average similar depths, and decrease confidence if depths do not agree
        // (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
        for (int i=0; i<sizeRef.height; ++i)
        {
            for (int j=0; j<sizeRef.width; ++j)
            {
                const ImageRef xRef(j,i);
                const MVS::Depth depth(depthDataRef.depthMap(xRef));
                if (depth == 0)
                {
                    newDepthMap(xRef) = 0;
                    newConfMap(xRef) = 0;
                    continue;
                }
                ASSERT(depth > 0);
                // update best depth and confidence estimate with all estimates
                float posConf(depthDataRef.confMap(xRef)), negConf(0);
                MVS::Depth avgDepth(depth*posConf);
                unsigned nPosViews(0), nNegViews(0);
                unsigned n(N);
                do
                {
                    const MVS::Depth d(depthMaps[--n](xRef));
                    if (d == 0)
                    {
                        if (nPosViews + nNegViews + n < nMinViews)
                            goto DiscardDepth;
                        continue;
                    }
                    ASSERT(d > 0);
                    if (IsDepthSimilar(depth, d, thDepthDiff))
                    {
                        // average similar depths
                        const float c(confMaps[n](xRef));
                        avgDepth += d*c;
                        posConf += c;
                        ++nPosViews;
                    } else
                    {
                        // penalize confidence
                        if (depth > d)
                        {
                            // occlusion
                            negConf += confMaps[n](xRef);
                        } else
                        {
                            // free-space violation
                            const auto& depthData = arrDepthData[depthDataRef.neighbors[idxNeighbors[n]].idx.ID];
                            const auto& camera = depthData.images.First().camera;
                            const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
                            const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
                            if (depthData.confMap.isInside(x))
                            {
                                const float c(depthData.confMap(x));
                                negConf += (c > 0 ? c : confMaps[n](xRef));
                            } else
                                negConf += confMaps[n](xRef);
                        }
                        ++nNegViews;
                    }
                } while (n);
                ASSERT(nPosViews+nNegViews >= nMinViews);
                // if enough good views and positive confidence...
                if (nPosViews >= nMinViewsAdjust && posConf > negConf && ISINSIDE(avgDepth/=posConf, depthDataRef.dMin, depthDataRef.dMax))
                {
                    // consider this pixel an inlier
                    newDepthMap(xRef) = avgDepth;
                    newConfMap(xRef) = posConf - negConf;
                } else
                {
                    // consider this pixel an outlier
DiscardDepth:
                    newDepthMap(xRef) = 0;
                    newConfMap(xRef) = 0;
                }
            }
        }
    } else
    {
        // remove depth if it does not agree with enough neighbors
        const float thDepthDiffStrict(MVS::OPTDENSE::fDepthDiffThreshold*0.8f);
        const unsigned nMinGoodViewsProc(75), nMinGoodViewsDeltaProc(65);
        const unsigned nDeltas(4);
        const unsigned nMinViewsDelta(nMinViews*(nDeltas-2));
        const ImageRef xDs[nDeltas] = { ImageRef(-1,0), ImageRef(1,0), ImageRef(0,-1), ImageRef(0,1) };
        for (int i=0; i<sizeRef.height; ++i)
        {
            for (int j=0; j<sizeRef.width; ++j)
            {
                const ImageRef xRef(j,i);
                const MVS::Depth depth(depthDataRef.depthMap(xRef));
                if (depth == 0)
                {
                    newDepthMap(xRef) = 0;
                    newConfMap(xRef) = 0;
                    continue;
                }
                ASSERT(depth > 0);
                // check if very similar with the neighbors projected to this pixel
                {
                    unsigned nGoodViews(0);
                    unsigned nViews(0);
                    unsigned n(N);
                    do
                    {
                        const MVS::Depth d(depthMaps[--n](xRef));
                        if (d > 0)
                        {
                            // valid view
                            ++nViews;
                            if (IsDepthSimilar(depth, d, thDepthDiffStrict))
                            {
                                // agrees with this neighbor
                                ++nGoodViews;
                            }
                        }
                    } while (n);
                    if (nGoodViews < nMinViews || nGoodViews < nViews*nMinGoodViewsProc/100)
                    {
                        newDepthMap(xRef) = 0;
                        newConfMap(xRef) = 0;
                        continue;
                    }
                }
                // check if similar with the neighbors projected around this pixel
                {
                    unsigned nGoodViews(0);
                    unsigned nViews(0);
                    for (unsigned d=0; d<nDeltas; ++d)
                    {
                        const ImageRef xDRef(xRef+xDs[d]);
                        unsigned n(N);
                        do
                        {
                            const MVS::Depth d(depthMaps[--n](xDRef));
                            if (d > 0)
                            {
                                // valid view
                                ++nViews;
                                if (IsDepthSimilar(depth, d, thDepthDiff))
                                {
                                    // agrees with this neighbor
                                    ++nGoodViews;
                                }
                            }
                        } while (n);
                    }
                    if (nGoodViews < nMinViewsDelta || nGoodViews < nViews*nMinGoodViewsDeltaProc/100)
                    {
                        newDepthMap(xRef) = 0;
                        newConfMap(xRef) = 0;
                        continue;
                    }
                }
                // enough good views, keep it
                newDepthMap(xRef) = depth;
                newConfMap(xRef) = depthDataRef.confMap(xRef);
            }
        }
    }
    return true;
}

// convert the ZNCC score to a weight used to average the fused points
inline float Conf2Weight(float conf, MVS::Depth depth)
{
    return 1.f/(MAXF(1.f-conf,0.03f)*depth*depth);
}

void DepthMapsData::FuseDepthMaps(MVS::PointCloud &pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
    TD_TIMER_STARTD();

    struct Proj
    {
        union
        {
            uint32_t idxPixel;
            struct
            {
                uint16_t x, y; // image pixel coordinates
            };
        };
        inline Proj() {}
        inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
        inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
        inline ImageRef GetCoord() const
        {
            return ImageRef(x,y);
        }
    };
    typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
    typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;

    // find best connected images
    IndexScoreArr connections(0, scene.images.GetSize());
    size_t nPointsEstimate(0);
    bool bNormalMap(true);
    FOREACH(i, scene.images)
    {
        MVS::DepthData& depthData = arrDepthData[i];
        if (!depthData.IsValid())
            continue;
        if(depthData.IsEmpty())
            continue;
        IndexScore& connection = connections.AddEmpty();
        connection.idx = i;
        connection.score = (float)scene.images[i].neighbors.GetSize();
        nPointsEstimate += ROUND2INT(depthData.depthMap.area()*(0.5f/*valid*/*0.3f/*new*/));
        if (depthData.normalMap.empty())
            bNormalMap = false;
    }
    connections.Sort();

    // fuse all depth-maps, processing the best connected images first
    const unsigned nMinViewsFuse(MINF(MVS::OPTDENSE::nMinViewsFuse, scene.images.GetSize()));
    const float normalError(COS(FD2R(MVS::OPTDENSE::fNormalDiffThreshold)));
    CLISTDEF0(MVS::Depth*) invalidDepths(0, 32);
    size_t nDepths(0);
    typedef TImage<cuint32_t> DepthIndex;
    typedef cList<DepthIndex> DepthIndexArr;
    DepthIndexArr arrDepthIdx(scene.images.GetSize());
    ProjsArr projs(0, nPointsEstimate);
    if (bEstimateNormal && !bNormalMap)
        bEstimateNormal = false;
    pointcloud.points.Reserve(nPointsEstimate);
    pointcloud.pointViews.Reserve(nPointsEstimate);
    pointcloud.pointWeights.Reserve(nPointsEstimate);
    if (bEstimateColor)
        pointcloud.colors.Reserve(nPointsEstimate);
    if (bEstimateNormal)
        pointcloud.normals.Reserve(nPointsEstimate);
    Util::Progress progress(_T("Fused depth-maps"), connections.GetSize());
    GET_LOGCONSOLE().Pause();
    FOREACHPTR(pConnection, connections)
    {
        TD_TIMER_STARTD();
        const uint32_t idxImage(pConnection->idx);
        const MVS::DepthData& depthData(arrDepthData[idxImage]);
        ASSERT(!depthData.images.IsEmpty() && !depthData.neighbors.IsEmpty());
        for (const auto& neighbor: depthData.neighbors)
        {
            DepthIndex& depthIdxs = arrDepthIdx[neighbor.idx.ID];
            if (!depthIdxs.empty())
                continue;
            const MVS::DepthData& depthDataB(arrDepthData[neighbor.idx.ID]);
            if (depthDataB.IsEmpty())
                continue;
            depthIdxs.create(depthDataB.depthMap.size());
            depthIdxs.memset((uint8_t)NO_ID);
        }
        ASSERT(!depthData.IsEmpty());
        const Image8U::Size sizeMap(depthData.depthMap.size());
        const auto& imageData = *depthData.images.First().pImageData;
        ASSERT(&imageData-scene.images.Begin() == idxImage);
        DepthIndex& depthIdxs = arrDepthIdx[idxImage];
        if (depthIdxs.empty())
        {
            depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
            depthIdxs.memset((uint8_t)NO_ID);
        }
        const size_t nNumPointsPrev(pointcloud.points.GetSize());
        for (int i=0; i<sizeMap.height; ++i)
        {
            for (int j=0; j<sizeMap.width; ++j)
            {
                const ImageRef x(j,i);
                const auto depth(depthData.depthMap(x));
                if (depth == 0)
                    continue;
                ++nDepths;
                ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
                uint32_t& idxPoint = depthIdxs(x);
                if (idxPoint != NO_ID)
                    continue;
                // create the corresponding 3D point
                idxPoint = (uint32_t)pointcloud.points.GetSize();
                auto& point = pointcloud.points.AddEmpty();
                point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
                auto& views = pointcloud.pointViews.AddEmpty();
                views.Insert(idxImage);
                auto& weights = pointcloud.pointWeights.AddEmpty();
                REAL confidence(weights.emplace_back(Conf2Weight(depthData.confMap(x),depth)));
                ProjArr& pointProjs = projs.AddEmpty();
                pointProjs.Insert(Proj(x));
                const MVS::PointCloud::Normal normal(bNormalMap ? Cast<MVS::Normal::Type>(imageData.camera.R.t()*Cast<REAL>(depthData.normalMap(x))) : MVS::Normal(0,0,-1));
                //ASSERT(ISEQUAL(norm(normal), 1.f));
                if(!(norm(normal), 1.f))
                {
                    std::cout << "WARNING from DepthMapsData::FuseDepthMaps: norm(normal) = " << norm(normal) << " != 1" << std::endl;
                }
                // check the projection in the neighbor depth-maps
                Point3 X(point*confidence);
                Pixel32F C(Cast<float>(imageData.image(x))*confidence);
                MVS::PointCloud::Normal N(normal*confidence);
                invalidDepths.Empty();
                FOREACHPTR(pNeighbor, depthData.neighbors) {
                    const uint32_t idxImageB(pNeighbor->idx.ID);
                    auto& depthDataB = arrDepthData[idxImageB];
                    if (depthDataB.IsEmpty())
                        continue;
                    const auto& imageDataB = scene.images[idxImageB];
                    const Point3f pt(imageDataB.camera.ProjectPointP3(point));
                    if (pt.z <= 0)
                        continue;
                    const ImageRef xB(ROUND2INT(pt.x/pt.z), ROUND2INT(pt.y/pt.z));
                    auto& depthMapB = depthDataB.depthMap;
                    if (!depthMapB.isInside(xB))
                        continue;
                    auto& depthB = depthMapB(xB);
                    if (depthB == 0)
                        continue;
                    uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
                    if (idxPointB != NO_ID)
                        continue;
                    if (IsDepthSimilar(pt.z, depthB, MVS::OPTDENSE::fDepthDiffThreshold))
                    {
                        // check if normals agree
                        const MVS::PointCloud::Normal normalB(bNormalMap ? Cast<MVS::Normal::Type>(imageDataB.camera.R.t()*Cast<REAL>(depthDataB.normalMap(xB))) : MVS::Normal(0,0,-1));
                        //ASSERT(ISEQUAL(norm(normalB), 1.f));
                        if(!(ISEQUAL(norm(normalB), 1.f)))
                        {
                            std::cout << "WARNING from DepthMapsData::FuseDepthMaps: norm(normalB) = " << norm(normalB) << " != 1" << std::endl;
                        }
                        if (normal.dot(normalB) > normalError)
                        {
                            // add view to the 3D point
                            ASSERT(views.FindFirst(idxImageB) == MVS::PointCloud::ViewArr::NO_INDEX);
                            const float confidenceB(Conf2Weight(depthDataB.confMap(xB),depthB));
                            const uint32_t idx(views.InsertSort(idxImageB));
                            weights.InsertAt(idx, confidenceB);
                            pointProjs.InsertAt(idx, Proj(xB));
                            idxPointB = idxPoint;
                            X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
                            if (bEstimateColor)
                                C += Cast<float>(imageDataB.image(xB))*confidenceB;
                            if (bEstimateNormal)
                                N += normalB*confidenceB;
                            confidence += confidenceB;
                            continue;
                        }
                    }
                    if (pt.z < depthB)
                    {
                        // discard depth
                        invalidDepths.Insert(&depthB);
                    }
                }
                if (views.GetSize() < nMinViewsFuse)
                {
                    // remove point
                    FOREACH(v, views)
                    {
                        const uint32_t idxImageB(views[v]);
                        const ImageRef x(pointProjs[v].GetCoord());
                        ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
                        arrDepthIdx[idxImageB](x).idx = NO_ID;
                    }
                    projs.RemoveLast();
                    pointcloud.pointWeights.RemoveLast();
                    pointcloud.pointViews.RemoveLast();
                    pointcloud.points.RemoveLast();
                } else
                {
                    // this point is valid, store it
                    const REAL nrm(REAL(1)/confidence);
                    point = X*nrm;
                    ASSERT(ISFINITE(point));
                    if (bEstimateColor)
                        pointcloud.colors.AddConstruct((C*(float)nrm).cast<uint8_t>());
                    if (bEstimateNormal)
                        pointcloud.normals.AddConstruct(normalized(N*(float)nrm));
                    // invalidate all neighbor depths that do not agree with it
                    for (auto* pDepth: invalidDepths)
                        *pDepth = 0;
                }
            }
        }
        ASSERT(pointcloud.points.GetSize() == pointcloud.pointViews.GetSize() && pointcloud.points.GetSize() == pointcloud.pointWeights.GetSize() && pointcloud.points.GetSize() == projs.GetSize());
        DEBUG_ULTIMATE("Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.GetSize()-1, pointcloud.points.GetSize()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
        progress.display(pConnection-connections.Begin());
    }
    GET_LOGCONSOLE().Play();
    progress.close();
    arrDepthIdx.Release();

    DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)", connections.GetSize(), nDepths, pointcloud.points.GetSize(), ROUND2INT((100.f*pointcloud.points.GetSize())/nDepths), TD_TIMER_GET_FMT().c_str());

    if (bEstimateNormal && !pointcloud.points.IsEmpty() && pointcloud.normals.IsEmpty())
    {
        // estimate normal also if requested (quite expensive if normal-maps not available)
        TD_TIMER_STARTD();
        pointcloud.normals.Resize(pointcloud.points.GetSize());
        const int64_t nPoints((int64_t)pointcloud.points.GetSize());
#pragma omp parallel for
        for (int64_t i=0; i< nPoints; ++i)
        {
            auto& weights = pointcloud.pointWeights[i];
            ASSERT(!weights.IsEmpty());
            uint32_t idxView(0);
            float bestWeight = weights.First();
            for (uint32_t idx=1; idx < weights.GetSize(); ++idx)
            {
                const auto& weight = weights[idx];
                if (bestWeight < weight)
                {
                    bestWeight = weight;
                    idxView = idx;
                }
            }
            const MVS::DepthData& depthData(arrDepthData[pointcloud.pointViews[i][idxView]]);
            ASSERT(depthData.IsValid() && !depthData.IsEmpty());
            depthData.GetNormal(projs[i][idxView].GetCoord(), pointcloud.normals[i]);
        }
        DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.points.GetSize(), TD_TIMER_GET_FMT().c_str());
    }
}
