#include "DepthEstimator.h"

const int DepthEstimator::nSizeHalfWindow = 5 ;
const int DepthEstimator::nSizeWindow = nSizeHalfWindow * 2 + 1 ;
const int DepthEstimator::nSizeStep = 2 ;
const int DepthEstimator::TexelChannels = 1 ;
const int DepthEstimator::nTexels = ((nSizeHalfWindow * 2 + nSizeStep) / nSizeStep) * ((nSizeHalfWindow * 2 + nSizeStep) / nSizeStep) * TexelChannels;


const float DepthEstimator::scaleRanges[12] = {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f};

DepthEstimator::DepthEstimator(unsigned nIter, MVS::DepthData& _depthData0, volatile Thread::safe_t& _idx, MVS::DepthEstimator::WeightMap& _weightMap0, const CLISTDEF0(TPoint2<uint16_t>)& _coords)
        :
      rnd(SEACAVE::Random::default_seed),
      idxPixel(_idx),
      neighbors(0,2),

      neighborsClose(0,4),

      scores(_depthData0.images.size()-1),
      depthMap0(_depthData0.depthMap), normalMap0(_depthData0.normalMap), confMap0(_depthData0.confMap),

      weightMap0(_weightMap0),

      nIteration(nIter),
      images(InitImages(_depthData0)), image0(_depthData0.images[0]),
      coords(_coords), size(_depthData0.images.First().image.size()),
      dMin(_depthData0.dMin), dMax(_depthData0.dMax),
      dMinSqr(SQRT(_depthData0.dMin)), dMaxSqr(SQRT(_depthData0.dMax)),
      dir(nIter % 2 ? RB2LT : LT2RB),

      idxScore(_depthData0.images.size()<=2 ? 0u : 1u),

      smoothBonusDepth(1.f - MVS::OPTDENSE::fRandomSmoothBonus), smoothBonusNormal((1.f - MVS::OPTDENSE::fRandomSmoothBonus) * 0.96f),
      smoothSigmaDepth(-1.f / (2.f * MVS::OPTDENSE::fRandomSmoothDepth * MVS::OPTDENSE::fRandomSmoothDepth)), // used in exp(-x^2 / (2*(0.02^2)))
      smoothSigmaNormal(-1.f / (2.f * FD2R(MVS::OPTDENSE::fRandomSmoothNormal) * FD2R(MVS::OPTDENSE::fRandomSmoothNormal))), // used in exp(-x^2 / (2*(0.22^2)))
      thMagnitudeSq(MVS::OPTDENSE::fDescriptorMinMagnitudeThreshold > 0 ? MVS::OPTDENSE::fDescriptorMinMagnitudeThreshold * MVS::OPTDENSE::fDescriptorMinMagnitudeThreshold : -1.f),
      angle1Range(FD2R(MVS::OPTDENSE::fRandomAngle1Range)),
      angle2Range(FD2R(MVS::OPTDENSE::fRandomAngle2Range)),
      thConfSmall(MVS::OPTDENSE::fNCCThresholdKeep * 0.2f),
      thConfBig(MVS::OPTDENSE::fNCCThresholdKeep * 0.4f),
      thConfRand(MVS::OPTDENSE::fNCCThresholdKeep * 0.9f),
      thRobust(MVS::OPTDENSE::fNCCThresholdKeep * 1.2f)
{

}

bool DepthEstimator::PreparePixelPatch(const ImageRef& x)
{
    x0 = x;
    return image0.image.isInside(ImageRef(x.x - nSizeHalfWindow, x.y - nSizeHalfWindow)) &&
            image0.image.isInside(ImageRef(x.x + nSizeHalfWindow, x.y + nSizeHalfWindow));
}

bool DepthEstimator::FillPixelPatch()
{
    auto& w = weightMap0[x0.y*image0.image.width()+x0.x];
    if (w.normSq0 == 0)
    {
        w.sumWeights = 0;
        int n = 0;
        const float colCenter = image0.image(x0);
        for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep)
        {
            for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep)
            {
                auto& pw = w.weights[n++];
                w.normSq0 +=
                        (pw.tempWeight = image0.image(x0.y+i, x0.x+j)) *
                        (pw.weight = GetWeight(ImageRef(j,i), colCenter));
                w.sumWeights += pw.weight;
            }
        }
        ASSERT(n == nTexels);
        const float tm(w.normSq0/w.sumWeights);
        w.normSq0 = 0;
        n = 0;
        do
        {
            auto& pw = w.weights[n];
            const float t(pw.tempWeight - tm);
            w.normSq0 += (pw.tempWeight = pw.weight * t) * t;
        } while (++n < nTexels);
    }
    normSq0 = w.normSq0;

    if (normSq0 < thMagnitudeSq)
        return false;
    reinterpret_cast<Point3&>(X0) = image0.camera.TransformPointI2C(Cast<REAL>(x0));
    return true;
}

float DepthEstimator::ScorePixelImage(const ViewData &image1, MVS::Depth depth, const MVS::Normal& normal)
{
    // center a patch of given size on the segment and fetch the pixel values in the target image
    Matrix3x3f H(ComputeHomographyMatrix(image1, depth, normal));
    Point3f X;
    ProjectVertex_3x3_2_3(H.val, Point2f(float(x0.x-nSizeHalfWindow),float(x0.y-nSizeHalfWindow)).ptr(), X.ptr());
    Point3f baseX(X);
    H *= float(nSizeStep);
    int n = 0;
    float sum = 0;

    float sumSq = 0, num = 0;


    const auto& w = weightMap0[x0.y*image0.image.width()+x0.x];

    for (int i=-nSizeHalfWindow; i<=nSizeHalfWindow; i+=nSizeStep)
    {
        for (int j=-nSizeHalfWindow; j<=nSizeHalfWindow; j+=nSizeStep)
        {
            const Point2f pt(X);
            if (!image1.view.image.isInsideWithBorder<float,1>(pt))
                return thRobust;
            const float v(image1.view.image.sample(pt));
            const auto& pw = w.weights[n++];
            const float vw(v*pw.weight);
            sum += vw;
            sumSq += v*vw;
            num += v*pw.tempWeight;
            X.x += H[0]; X.y += H[3]; X.z += H[6];
        }
        baseX.x += H[1]; baseX.y += H[4]; baseX.z += H[7];
        X = baseX;
    }
    ASSERT(n == nTexels);
    // score similarity of the reference and target texture patches

    const float normSq1(sumSq-SQUARE(sum)/w.sumWeights);

    const float nrmSq(normSq0*normSq1);
    if (nrmSq <= 0.f)
        return thRobust;

    const float ncc(CLAMP(num/SQRT(nrmSq), -1.f, 1.f));
    double score(1 - ncc);
    // encourage smoothness
    for (const NeighborEstimate& neighbor: neighborsClose)
    {
        const float factorDepth(DENSE_EXP(SQUARE(plane.Distance(neighbor.X)/depth) * smoothSigmaDepth));
        const float factorNormal(DENSE_EXP(SQUARE(ACOS(ComputeAngle<float,float>(normal.ptr(), neighbor.normal.ptr()))) * smoothSigmaNormal));
        score *= (1 - smoothBonusDepth * factorDepth) * (1 - smoothBonusNormal * factorNormal);
    }

    //ASSERT(ISFINITE(score));
    if(!ISFINITE(score))
        std::cout << "WARNING from DepthEstimator::ScorePixelImage: " << score << " infinite" << std::endl;
    return score;
}

float DepthEstimator::ScorePixel(MVS::Depth depth, const MVS::Normal& normal)
{
    ASSERT(depth > 0 && normal.dot(Cast<float>(static_cast<const Point3&>(X0))) <= 0);
    // compute score for this pixel as seen in each view
    ASSERT(scores.size() == images.size());
    FOREACH(idxView, images)
            scores[idxView] = ScorePixelImage(images[idxView], depth, normal);
    // set score as the min-mean similarity
    if (idxScore == 0)
        return *std::min_element(scores.cbegin(), scores.cend());

    const float* pescore(&scores.GetNth(idxScore));
    const float* pscore(scores.cbegin());
    int n = 1;
    float score = *pscore;
    do {
        const float s(*(++pscore));
        if (s >= thRobust)
            break;
        score += s;
        ++n;
    } while (pscore < pescore);
    return score / n;
}

void DepthEstimator::ProcessPixel(IDX idx)
{
    // compute pixel coordinates from pixel index and its neighbors
    if (!PreparePixelPatch(dir == LT2RB ? coords[idx] : coords[coords.GetSize()- 1 -idx]) || !FillPixelPatch())
        return;

    // find neighbors
    neighbors.Empty();
    neighborsClose.Empty();
    if (dir == LT2RB)
    {
        // direction from left-top to right-bottom corner
        if (x0.x > nSizeHalfWindow)
        {
            const ImageRef nx(x0.x - 1, x0.y);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
            {
                neighbors.emplace_back(nx);
                neighborsClose.emplace_back(NeighborEstimate{ ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth))) });
            }
        }
        if (x0.y > nSizeHalfWindow)
        {
            const ImageRef nx(x0.x, x0.y - 1);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
            {
                neighbors.emplace_back(nx);
                neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))});
            }
        }
        if (x0.x < size.width - nSizeHalfWindow)
        {
            const ImageRef nx(x0.x+1, x0.y);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
                neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))});
        }
        if (x0.y < size.height - nSizeHalfWindow)
        {
            const ImageRef nx(x0.x, x0.y+1);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
                neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))});
        }
    } else
    {
        // direction from right-bottom to left-top corner
        if (x0.x < size.width - nSizeHalfWindow)
        {
            const ImageRef nx(x0.x+1, x0.y);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
            {
                neighbors.emplace_back(nx);
                neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth))) });
            }
        }
        if (x0.y < size.height-nSizeHalfWindow)
        {
            const ImageRef nx(x0.x, x0.y+1);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
            {
                neighbors.emplace_back(nx);
                neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))});
            }
        }

        if (x0.x > nSizeHalfWindow)
        {
            const ImageRef nx(x0.x-1, x0.y);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
                neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))});
        }
        if (x0.y > nSizeHalfWindow)
        {
            const ImageRef nx(x0.x, x0.y-1);
            const MVS::Depth ndepth(depthMap0(nx));
            if (ndepth > 0)
                neighborsClose.emplace_back(NeighborEstimate{ndepth,normalMap0(nx), Cast<float>(image0.camera.TransformPointI2C(Point3(nx, ndepth)))});
        }

    }
    float& conf = confMap0(x0);
    auto& depth = depthMap0(x0);
    auto& normal = normalMap0(x0);
    const MVS::Normal viewDir(Cast<float>(reinterpret_cast<const Point3&>(X0)));
    ASSERT(depth > 0 && normal.dot(viewDir) <= 0);

    // check if any of the neighbor estimates are better then the current estimate

    FOREACH(n, neighbors)
    {
        const ImageRef& nx = neighbors[n];

        if (confMap0(nx) >= MVS::OPTDENSE::fNCCThresholdKeep)
            continue;
        NeighborEstimate& neighbor = neighborsClose[n];
        neighbor.depth = InterpolatePixel(nx, neighbor.depth, neighbor.normal);
        CorrectNormal(neighbor.normal);
        //ASSERT(neighbor.depth > 0 && neighbor.normal.dot(viewDir) <= 0);
        if(!(neighbor.depth > 0 && neighbor.normal.dot(viewDir) <= 0))
        {
            std::cout << "WARNING from DepthEstimator::ProcessPixel: 540 str" << std::endl;
        }

        InitPlane(neighbor.depth, neighbor.normal);
        const float nconf(ScorePixel(neighbor.depth, neighbor.normal));
        //ASSERT(nconf >= 0 && nconf <= 2);
        if(!(nconf >= 0 && nconf <= 2))
        {
            std::cout << "WARNING from DepthEstimator::ProcessPixel: 548 str" << std::endl;
        }

        if (conf > nconf)
        {
            conf = nconf;
            depth = neighbor.depth;
            normal = neighbor.normal;
        }
    }

    // try random values around the current estimate in order to refine it

    unsigned idxScaleRange(0);
RefineIters:
    if (conf <= thConfSmall)
        idxScaleRange = 2;
    else if (conf <= thConfBig)
        idxScaleRange = 1;
    else if (conf >= thConfRand)
    {
        // try completely random values in order to find an initial estimate
        for (unsigned iter = 0; iter < MVS::OPTDENSE::nRandomIters; ++iter)
        {
            const MVS::Depth ndepth(RandomDepth(dMinSqr, dMaxSqr));
            const MVS::Normal nnormal(RandomNormal(viewDir));
            const float nconf(ScorePixel(ndepth, nnormal));
            //ASSERT(nconf >= 0);
            if (nconf < 0)
                std::cout << "WARNING from DepthEstimator::ProcessPixel: nconf = " << nconf << "< 0" << std::endl;
            if (conf > nconf)
            {
                conf = nconf;
                depth = ndepth;
                normal = nnormal;
                if (conf < thConfRand)
                    goto RefineIters;
            }
        }
        return;
    }
    float scaleRange(scaleRanges[idxScaleRange]);
    const float depthRange(MaxDepthDifference(depth, MVS::OPTDENSE::fRandomDepthRatio));
    Point2f p;
    Normal2Dir(normal, p);
    MVS::Normal nnormal;

    for (unsigned iter=0; iter < MVS::OPTDENSE::nRandomIters; ++iter)
    {
        const MVS::Depth ndepth(rnd.randomMeanRange(depth, depthRange * scaleRange));
        if (!ISINSIDE(ndepth, dMin, dMax))
            continue;
        const Point2f np(rnd.randomMeanRange(p.x, angle1Range * scaleRange), rnd.randomMeanRange(p.y, angle2Range * scaleRange));
        Dir2Normal(np, nnormal);
        if (nnormal.dot(viewDir) >= 0)
            continue;

        InitPlane(ndepth, nnormal);

        const float nconf(ScorePixel(ndepth, nnormal));
        ASSERT(nconf >= 0);
        if (conf > nconf)
        {
            conf = nconf;
            depth = ndepth;
            normal = nnormal;
            p = np;
            scaleRange = scaleRanges[++idxScaleRange];
        }
    }
}

void DepthEstimator::ProcessPixels(int _iterations)
{

}


MVS::Depth DepthEstimator::InterpolatePixel(const ImageRef& nx, MVS::Depth depth, const MVS::Normal& normal) const
{
    //ASSERT(depth > 0 && normal.dot(image0.camera.TransformPointI2C(Cast<REAL>(nx))) <= 0);
    if(!(depth > 0 && normal.dot(image0.camera.TransformPointI2C(Cast<REAL>(nx))) <= 0))
        std::cout << "WARNING from DepthEstimator::InterpolatePixel: depth = " << depth<< " dot = " << normal.dot(image0.camera.TransformPointI2C(Cast<REAL>(nx))) << std::endl;
    MVS::Depth depthNew;
    // compute as intersection of the lines
    // {(x1, y1), (x2, y2)} from neighbor's 3D point towards normal direction
    // and
    // {(0, 0), (x4, 1)} from camera center towards current pixel direction
    // in the x or y plane
    if (x0.x == nx.x)
    {
        const float fy = (float)image0.camera.K[4];
        const float cy = (float)image0.camera.K[5];
        const float x1 = depth * (nx.y - cy) / fy;
        const float y1 = depth;
        const float x4 = (x0.y - cy) / fy;
        const float denom = normal.z + x4 * normal.y;
        if (ISZERO(denom))
            return depth;
        const float x2 = x1 + normal.z;
        const float y2 = y1 - normal.y;
        const float nom = y1 * x2 - x1 * y2;
        depthNew = nom / denom;
    }
    else
    {
        ASSERT(x0.y == nx.y);
        const float fx = (float)image0.camera.K[0];
        const float cx = (float)image0.camera.K[2];
        ASSERT(image0.camera.K[1] == 0);
        const float x1 = depth * (nx.x - cx) / fx;
        const float y1 = depth;
        const float x4 = (x0.x - cx) / fx;
        const float denom = normal.z + x4 * normal.x;
        if (ISZERO(denom))
            return depth;
        const float x2 = x1 + normal.z;
        const float y2 = y1 - normal.x;
        const float nom = y1 * x2 - x1 * y2;
        depthNew = nom / denom;
    }
    return ISINSIDE(depthNew,dMin,dMax) ? depthNew : depth;
}

void DepthEstimator::InitPlane(MVS::Depth depth, const MVS::Normal& normal)
{
    plane.m_vN = reinterpret_cast<const Vec3f&>(normal);
    plane.m_fD = -depth*reinterpret_cast<const Vec3f&>(normal).dot(Cast<float>(X0));
}

float DepthEstimator::GetWeight(const ImageRef &x, float center) const
{
    // color weight [0..1]
    const float sigmaColor(-1.f/(2.f*SQUARE(0.2f)));
    const float wColor(SQUARE(image0.image(x0+x)-center) * sigmaColor);
    // spatial weight [0..1]
    const float sigmaSpatial(-1.f/(2.f*SQUARE((int)nSizeHalfWindow)));
    const float wSpatial(float(SQUARE(x.x) + SQUARE(x.y)) * sigmaSpatial);
    return DENSE_EXP(wColor+wSpatial);
}

Matrix3x3f DepthEstimator::ComputeHomographyMatrix(const ViewData &img, MVS::Depth depth, const MVS::Normal &normal) const
{
    // compute homography matrix as above, caching some constants
    const Vec3 n(normal);
    return (img.Hl + img.Hm * (n.t()*INVERT(n.dot(X0)*depth))) * img.Hr;
}

CLISTDEF0IDX(DepthEstimator::ViewData, MVS::IIndex) DepthEstimator::InitImages(const MVS::DepthData &depthData)
{
    CLISTDEF0IDX(ViewData, MVS::IIndex) images(0, depthData.images.GetSize() - 1);
    const MVS::DepthData::ViewData& image0(depthData.images.First());
    for (MVS::IIndex i=1; i< depthData.images.GetSize(); ++i)
        images.AddConstruct(image0, depthData.images[i]);
    return images;
}

Point3 DepthEstimator::ComputeRelativeC(const MVS::DepthData &depthData)
{
    return depthData.images[1].camera.R * (depthData.images[0].camera.C-depthData.images[1].camera.C);
}

Matrix3x3 DepthEstimator::ComputeRelativeR(const MVS::DepthData &depthData)
{
    RMatrix R;
    ComputeRelativeRotation(depthData.images[0].camera.R, depthData.images[1].camera.R, R);
    return std::move(R);
}

MVS::Depth DepthEstimator::RandomDepth(MVS::Depth dMinSqr, MVS::Depth dMaxSqr)
{
    ASSERT(dMinSqr > 0 && dMinSqr < dMaxSqr);
    return SQUARE(rnd.randomRange(dMinSqr, dMaxSqr));
}

MVS::Normal DepthEstimator::RandomNormal(const Point3f &viewRay)
{
    MVS::Normal normal;
    Dir2Normal(Point2f(rnd.randomRange(FD2R(0.f),FD2R(180.f)), rnd.randomRange(FD2R(90.f), FD2R(180.f))), normal);
    return normal.dot(viewRay) > 0 ? -normal : normal;
}

void DepthEstimator::CorrectNormal(MVS::Normal &normal) const
{
    const MVS::Normal viewDir(Cast<float>(X0));
    const float cosAngLen(normal.dot(viewDir));
    if (cosAngLen >= 0)
        normal = RMatrixBaseF(normal.cross(viewDir), MINF((ACOS(cosAngLen/norm(viewDir))-FD2R(90.f)) * 1.01f, -0.001f)) * normal;
}
