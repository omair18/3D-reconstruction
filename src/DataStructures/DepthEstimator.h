#ifndef DEPTH_ESTIMATOR_H
#define DEPTH_ESTIMATOR_H
#include <OpenMVS/MVS.h>

class DepthEstimator
{ 
public:
    struct ViewData
    {
        const MVS::DepthData::ViewData& view;
        const Matrix3x3 Hl;   //
        const Vec3 Hm;	      // constants during per-pixel loops
        const Matrix3x3 Hr;   //
        inline ViewData() : view(*((const MVS::DepthData::ViewData*)this)) {}
        inline ViewData(const MVS::DepthData::ViewData& image0, const MVS::DepthData::ViewData& image1)
            : view(image1),
              Hl(image1.camera.K * image1.camera.R * image0.camera.R.t()),
              Hm(image1.camera.K * image1.camera.R * (image0.camera.C - image1.camera.C)),
              Hr(image0.camera.K.inv()) {}
    };

    struct NeighborEstimate
    {
        MVS::Depth depth;
        MVS::Normal normal;
        Planef::POINT X;
    };

    enum ENDIRECTION
    {
        LT2RB = 0,
        RB2LT
    };

    static const int nSizeHalfWindow;
    static const int nSizeWindow;
    static const int  nSizeStep;
    static const int TexelChannels;
    static const int nTexels;

    SEACAVE::Random rnd;

    volatile Thread::safe_t& idxPixel; // current image index to be processed
    CLISTDEF0IDX(ImageRef, MVS::IIndex) neighbors; // neighbor pixels coordinates to be processed

    CLISTDEF0IDX(NeighborEstimate, MVS::IIndex) neighborsClose; // close neighbor pixel depths to be used for smoothing
    Vec3 X0;	      //
    ImageRef x0;	  // constants during one pixel loop
    float normSq0;	  //

    FloatArr scores;

    Planef plane; // plane defined by current depth and normal estimate

    MVS::DepthMap& depthMap0;
    MVS::NormalMap& normalMap0;
    MVS::ConfidenceMap& confMap0;

    MVS::DepthEstimator::WeightMap& weightMap0;

    const unsigned nIteration; // current PatchMatch iteration
    const CLISTDEF0IDX(ViewData, MVS::IIndex) images; // neighbor images used
    const MVS::DepthData::ViewData& image0;

    const CLISTDEF0(TPoint2<uint16_t>)& coords;
    const Image8U::Size size;
    const MVS::Depth dMin, dMax;
    const MVS::Depth dMinSqr, dMaxSqr;
    const ENDIRECTION dir;

    const IDX idxScore;

    DepthEstimator(unsigned nIter, MVS::DepthData& _depthData0, volatile Thread::safe_t& _idx, MVS::DepthEstimator::WeightMap& _weightMap0, const CLISTDEF0(TPoint2<uint16_t>)& _coords);

    bool PreparePixelPatch(const ImageRef&);
    bool FillPixelPatch();
    float ScorePixelImage(const ViewData& image1, MVS::Depth, const MVS::Normal&);
    float ScorePixel(MVS::Depth, const MVS::Normal&);
    void ProcessPixel(IDX idx);
    void ProcessPixels(int iterations);
    MVS::Depth InterpolatePixel(const ImageRef&, MVS::Depth, const MVS::Normal&) const;
    void InitPlane(MVS::Depth, const MVS::Normal&);

    float GetWeight(const ImageRef& x, float center) const;

    Matrix3x3f ComputeHomographyMatrix(const ViewData& img, MVS::Depth depth, const MVS::Normal& normal) const;

    static CLISTDEF0IDX(ViewData, MVS::IIndex) InitImages(const MVS::DepthData& depthData);

    static Point3 ComputeRelativeC(const MVS::DepthData& depthData);

    static Matrix3x3 ComputeRelativeR(const MVS::DepthData& depthData);

    // generate random depth and normal
    MVS::Depth RandomDepth(MVS::Depth dMinSqr, MVS::Depth dMaxSqr);
    MVS::Normal RandomNormal(const Point3f& viewRay);

    // adjust normal such that it makes at most 90 degrees with the viewing angle
    inline void CorrectNormal(MVS::Normal& normal) const;

    static void MapMatrix2ZigzagIdx(const Image8U::Size& size, CLISTDEF0(TPoint2<uint16_t>)& coords, const BitMatrix& mask, int rawStride = 16);

    const float smoothBonusDepth, smoothBonusNormal;
    const float smoothSigmaDepth, smoothSigmaNormal;
    const float thMagnitudeSq;
    const float angle1Range, angle2Range;
    const float thConfSmall, thConfBig, thConfRand;
    const float thRobust;

    // replace POWI(0.5f, (int)idxScaleRange):
    static const float scaleRanges[12];


};
#endif //DEPTH_ESTIMATOR_H
