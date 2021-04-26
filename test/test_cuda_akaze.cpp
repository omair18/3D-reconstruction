#include <nppi_filtering_functions.h>
#include <nppi_statistics_functions.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <map>

#include "cuda_test_kernels.h"
#include "CUDAImage.h"
#include "Logger.h"

/*
//
enum DESCRIPTOR_TYPE
{
    SURF_UPRIGHT = 0, ///< Upright descriptors, not invariant to rotation
    SURF = 1,
    MSURF_UPRIGHT = 2, ///< Upright descriptors, not invariant to rotation
    MSURF = 3,
    MLDB_UPRIGHT = 4, ///< Upright descriptors, not invariant to rotation
    MLDB = 5
};

enum DIFFUSIVITY_TYPE
{
    PM_G1 = 0,
    PM_G2 = 1,
    WEICKERT = 2,
    CHARBONNIER = 3
};

struct AKAZEOptions
{
    int omin;                                                 ///< Initial octave level (-1 means that the size of the input image is duplicated)
    int omax = 4;                                             ///< Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    int nsublevels = 4;                                       ///< Default number of sublevels per scale level
    int img_width;                                            ///< Width of the input image
    int img_height;                                           ///< Height of the input image
    float soffset = 1.6f;                                     ///< Base scale offset (sigma units)
    float derivative_factor = 1.5f;                           ///< Factor for the multiscale derivatives
    float sderivatives = 1.0;                                 ///< Smoothing factor for the derivatives
    DIFFUSIVITY_TYPE diffusivity = DIFFUSIVITY_TYPE::PM_G2;   ///< Diffusivity type

    float dthreshold = 0.001f;                                ///< Detector response threshold to accept point
    float min_dthreshold = 0.00001f;                          ///< Minimum detector threshold to accept a point

    DESCRIPTOR_TYPE descriptor = DESCRIPTOR_TYPE::MLDB;       ///< Type of descriptor
    int descriptor_size = 0;                                  ///< Size of the descriptor in bits. 0->Full size
    int descriptor_channels = 3;                              ///< Number of channels in the descriptor (1, 2, 3)
    int descriptor_pattern_size = 10;                         ///< Actual patch size is 2*pattern_size*point.scale

    float kcontrast = 0.001f;                                 ///< The contrast factor parameter
    float kcontrast_percentile = 0.7f;                        ///< Percentile level for the contrast factor
    size_t kcontrast_nbins = 300;                             ///< Number of bins for the contrast factor histogram

    int ncudaimages = 4;                                      ///< Number of CUDA images allocated per octave
    int maxkeypoints = 16*8192;                               ///< Maximum number of keypoints allocated
};

/// AKAZE Timing structure
struct AKAZETiming
{
    double kcontrast = 0.0;       ///< Contrast factor computation time in ms
    double scale = 0.0;           ///< Nonlinear scale space computation time in ms
    double derivatives = 0.0;     ///< Multiscale derivatives computation time in ms
    double detector = 0.0;        ///< Feature detector computation time in ms
    double extrema = 0.0;         ///< Scale space extrema computation time in ms
    double subpixel = 0.0;        ///< Subpixel refinement computation time in ms
    double descriptor = 0.0;      ///< Descriptors computation time in ms
};

struct TEvolution
{
    DataStructures::CUDAImage Lx, Ly;                   ///< First order spatial derivatives
    DataStructures::CUDAImage Lxx, Lxy, Lyy;            ///< Second order spatial derivatives
    DataStructures::CUDAImage Lflow;                    ///< Diffusivity image
    DataStructures::CUDAImage Lt;                       ///< Evolution image
    DataStructures::CUDAImage Lsmooth;                  ///< Smoothed image
    DataStructures::CUDAImage Lstep;                    ///< Evolution step update
    DataStructures::CUDAImage Ldet;                     ///< Detector response
    float etime = 0.0f;                                 ///< Evolution time
    float esigma = 0.0f;                                ///< Evolution sigma. For linear diffusion t = sigma^2 / 2
    size_t octave = 0;                                  ///< Image octave
    size_t sublevel = 0;                                ///< Image sublevel in each octave
    size_t sigma_size = 0;                              ///< Integer sigma. For computing the feature detector responses
};
*/
/*
bool fed_is_prime_internal(const int number)
{
    if (number <= 1)
    {
        return false;
    }
    else if (number == 2 || number == 3 || number == 5 || number == 7)
    {
        return true;
    }
    else if ((number % 2) == 0 || (number % 3) == 0 || (number % 5) == 0 || (number % 7) == 0)
    {
        return false;
    }
    else
    {
        int upperLimit = sqrt(number+1.0);
        int divisor = 11;

        while (divisor <= upperLimit )
        {
            if (number % divisor == 0)
            {
                return false;
            }

            divisor +=2;
        }
        return true;
    }
}

int fed_tau_internal(const int n, const float scale, const float tau_max, const bool reordering, std::vector<float>& tau)
{

    float c = 0.0, d = 0.0;     // Time savers
    std::vector<float> tauh;    // Helper vector for unsorted taus

    if (n <= 0)
        return 0;

    // Allocate memory for the time step size
    tau = std::vector<float>(n);

    if (reordering)
        tauh = std::vector<float>(n);

    // Compute time saver
    c = 1.0f / (4.0f * (float)n + 2.0f);
    d = scale * tau_max / 2.0f;

    // Set up originally ordered tau vector
    for (int k = 0; k < n; ++k)
    {
        float h = cos(M_PI * (2.0f * (float)k + 1.0f) * c);

        if (reordering)
            tauh[k] = d / (h * h);
        else
            tau[k] = d / (h * h);
    }

    // Permute list of time steps according to chosen reordering function
    int kappa = 0, prime = 0;

    if (reordering == true)
    {
        // Choose kappa cycle with k = n/2
        // This is a heuristic. We can use Leja ordering instead!!
        kappa = n / 2;

        // Get modulus for permutation
        prime = n + 1;

        while (!fed_is_prime_internal(prime))
        {
            prime++;
        }

        // Perform permutation
        for (int k = 0, l = 0; l < n; ++k, ++l)
        {
            int index = 0;
            while ((index = ((k+1)*kappa) % prime - 1) >= n)
            {
                k++;
            }

            tau[l] = tauh[index];
        }
    }

    return n;
}

int fed_tau_by_cycle_time(const float t, const float tau_max, const bool reordering, std::vector<float>& tau)
{
    int n = 0;          // Number of time steps
    float scale = 0.0;  // Ratio of t we search to maximal t

    // Compute necessary number of time steps
    n = (int)(ceil(sqrt(3.0* t / tau_max + 0.25f) - 0.5f - 1.0e-8f) + 0.5f);
    scale = 3.0 * t / (tau_max * (float)(n * (n + 1)));

    // Call internal FED time step creation routine
    return fed_tau_internal(n,scale,tau_max,reordering,tau);
}

int fed_tau_by_process_time(const float T, const int M, const float tau_max, const bool reordering, std::vector<float>& tau)
{
    // All cycles have the same fraction of the stopping time
    return fed_tau_by_cycle_time(T / (float)M, tau_max, reordering, tau);
}
*/

struct Params
{
    int iNbOctave = 4; ///< Octave to process
    int iNbSlicePerOctave = 4; ///< Levels per octave
    float fSigma0 = 1.6f; ///< Initial sigma offset (used to suppress low level noise)
    float fThreshold = 0.0008f;  ///< Hessian determinant threshold
    float fDesc_factor = 1.f;   ///< Magnifier used to describe an interest point
};

void checkCudaErrors(cudaError_t&& error)
{
    if(error != cudaError_t::cudaSuccess)
    {
        std::clog << "CUDA error " << static_cast<int>(error) << ": " << cudaGetErrorName(error) << " - "
                  << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error("CUDA RTE.");
    }
}

void checkNppErrors(NppStatus&& error)
{
    if(error != NppStatus::NPP_NO_ERROR)
    {
        std::clog << "NPPI error " << static_cast<int>(error) << std::endl;
        throw std::runtime_error("CUDA NPPI RTE.");
    }
}

void ComputeGaussianKernel(std::vector<float>& kernel, float sigma, size_t kernelSize)
{
    if(kernelSize % 2 != 1 && kernelSize != 0)
    {
        throw std::runtime_error("123");
    }

    // If kernel size is 0 computes its size using uber formula
    size_t k_size = ( kernelSize == 0 ) ? ceil( 2.0 * ( 1.0 + ( sigma - 0.8 ) / ( 0.3 ) ) ) : kernelSize;

    // Make kernel odd width
    k_size = ( k_size % 2 == 0 ) ? k_size + 1 : k_size;
    const size_t half_k_size = ( k_size - 1 ) / 2;

    kernel.resize(k_size);

    const double exp_scale = 1.0 / ( 2.0 * sigma * sigma );

    // Compute unnormalized kernel
    double sum = 0.0;
    for (size_t i = 0; i < k_size; ++i )
    {
        const double dx = ( static_cast<double>( i ) - static_cast<double>( half_k_size ) );
        kernel[i] = exp( - dx * dx * exp_scale );
        sum += kernel[i];
    }

    // Normalize kernel
    const double inv = 1.0 / sum;
    for (size_t i = 0; i < k_size; ++i )
    {
        kernel[i] *= inv;
    }
}

/// Allocate memory
/*
    AKAZEOptions options_;
    options_.img_width = sourceImageWidth;
    options_.img_height = sourceImageHeight;
    std::vector<TEvolution> evolution_;         ///< Vector of nonlinear diffusion evolution

    /// FED parameters
    int ncycles_ = 0;                               ///< Number of cycles
    bool reordering_ = true;                           ///< Flag for reordering time steps
    std::vector<std::vector<float > > tsteps_;  ///< Vector of FED dynamic time steps
    std::vector<int> nsteps_;                   ///< Vector of number of steps per cycle

    float rfactor = 0.0;
    int level_height = 0, level_width = 0;

    // Allocate the dimension of the matrices for the evolution
    for (int i = 0; i <= options_.omax - 1; i++)
    {
        rfactor = 1.0 / pow(2.0f, i);
        level_height = (int)(options_.img_height * rfactor);
        level_width = (int)(options_.img_width * rfactor);

        // Smallest possible octave and allow one scale if the image is small
        if ((level_width < 80 || level_height < 40) && i != 0)
        {
            options_.omax = i;
            break;
        }

        for (int j = 0; j < options_.nsublevels; j++)
        {
            TEvolution step;
            cv::Size size(level_width, level_height);
            step.Lx.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Ly.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Lxx.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Lxy.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Lyy.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Lt.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Ldet.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Lflow.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Lstep.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);
            step.Lsmooth.Allocate(level_width, level_height, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);

            step.esigma = options_.soffset * pow(2.0f, (float)(j) / (float)(options_.nsublevels) + i);
            step.sigma_size = (int)(step.esigma+0.5f); /// round to nearest integer
            step.etime = 0.5 * (step.esigma * step.esigma);
            step.octave = i;
            step.sublevel = j;
            evolution_.push_back(std::move(step));
        }
    }

    // Allocate memory for the number of cycles and time steps
    for (size_t i = 1; i < evolution_.size(); i++)
    {
        int naux = 0;
        std::vector<float> tau;
        float ttime = 0.0;
        ttime = evolution_[i].etime - evolution_[i - 1].etime;
        float tmax = 0.25;// * (1 << 2 * evolution_[i].octave);
        naux = fed_tau_by_process_time(ttime, 1, tmax, reordering_, tau);
        nsteps_.push_back(naux);
        tsteps_.push_back(std::move(tau));
        ncycles_++;
    }
*/


int main()
{
    LOGGER_INIT();
    /// Prepare

    cv::Mat image1 = cv::imread("/home/valera/Photo/30/IMG_20201011_131521.jpg");
    //cv::imshow("test", image1);
    cv::waitKey(0);

    DataStructures::CUDAImage sourceImage;
    sourceImage.CopyFromCvMat(image1);

    checkCudaErrors(cudaDeviceSynchronize());

    cudaStream_t cudaStream;
    checkCudaErrors(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));


    /// Now we have BGR image on GPU.
    /// Grayscaling

    DataStructures::CUDAImage grayImage;
    grayImage.Allocate(sourceImage.width_, sourceImage.height_, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, true);
    std::vector<float> grayscaleCoefficients = {0.114f, 0.587f, 0.299f};

    NppiSize grayScaleRoi = { .width = (int)sourceImage.width_, .height = (int)sourceImage.height_ };

    checkNppErrors(nppiColorToGray_8u_C3C1R(sourceImage.gpuData_, sourceImage.pitch_, grayImage.gpuData_, grayImage.pitch_, grayScaleRoi, grayscaleCoefficients.data()));

    /// Converting to float image

    NppiSize floatConvertionRoi = grayScaleRoi;
    DataStructures::CUDAImage grayFloatImage;
    grayFloatImage.Allocate(grayImage.width_, grayImage.height_, grayImage.channels_, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);

    checkNppErrors(nppiConvert_8u32f_C1R(grayImage.gpuData_, grayImage.pitch_, (float *)grayFloatImage.gpuData_, grayFloatImage.pitch_, floatConvertionRoi));

    /// Normalize float image

    elementwise_divide_float_api(255., (float *)grayFloatImage.gpuData_, grayFloatImage.width_, grayFloatImage.height_, grayFloatImage.pitch_, grayFloatImage.channels_, cudaStream);

    checkCudaErrors(cudaStreamSynchronize(cudaStream));

    /// Compute contrast factor
    Params params;

    float contrastFactor = 0;
    const size_t nb_bin = 300;

    DataStructures::CUDAImage smoothedGrayFloatImage;
    DataStructures::CUDAImage tempImage;

    tempImage.Allocate(grayFloatImage.width_, grayFloatImage.height_, grayFloatImage.channels_, grayFloatImage.elementType_, grayFloatImage.pitchedAllocation_);
    smoothedGrayFloatImage.Allocate(grayFloatImage.width_, grayFloatImage.height_, grayFloatImage.channels_, grayFloatImage.elementType_, grayFloatImage.pitchedAllocation_);

    // map <(sigma, size), kernel>
    std::map<std::pair<float, size_t>, std::vector<float>> gaussKernelsHorizontal;
    std::map<std::pair<float, size_t>, std::vector<float>> gaussKernelsVertical;

    // compute gaussian kernel for separable convolution

    float gaussSmoothSigma = 1.f;
    size_t gaussSmoothKernelSize = 0; // automatic computation

    auto gaussSmoothKernelKey = std::make_pair(gaussSmoothSigma, gaussSmoothKernelSize);
    std::vector<float> horizontalGaussKernel;
    std::vector<float> verticalGaussKernel;

    ComputeGaussianKernel(verticalGaussKernel, gaussSmoothSigma, gaussSmoothKernelSize);
    ComputeGaussianKernel(horizontalGaussKernel, gaussSmoothSigma, gaussSmoothKernelSize);

    gaussKernelsHorizontal.insert(std::make_pair(gaussSmoothKernelKey, horizontalGaussKernel));
    gaussKernelsVertical.insert(std::make_pair(gaussSmoothKernelKey, verticalGaussKernel));

    //convolution step

    NppiPoint srcOffset {.x = 0, .y = 0};
    //prepare kernels
    DataStructures::CUDAImage verticalKernel;
    DataStructures::CUDAImage horizontalKernel;

    verticalKernel.CopyFromRawHostPointer(
            gaussKernelsVertical.at(gaussSmoothKernelKey).data(),
            gaussKernelsVertical.at(gaussSmoothKernelKey).size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    horizontalKernel.CopyFromRawHostPointer(
            gaussKernelsHorizontal.at(gaussSmoothKernelKey).data(),
            gaussKernelsHorizontal.at(gaussSmoothKernelKey).size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    checkNppErrors(nppiFilterRowBorder_32f_C1R((float*)grayFloatImage.gpuData_, grayFloatImage.pitch_, floatConvertionRoi, srcOffset, (float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, (float *)verticalKernel.gpuData_, verticalKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    checkNppErrors(nppiFilterColumnBorder_32f_C1R((float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, srcOffset, (float*)smoothedGrayFloatImage.gpuData_, smoothedGrayFloatImage.pitch_, floatConvertionRoi, (float*)horizontalKernel.gpuData_, horizontalKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    verticalKernel.Release();
    horizontalKernel.Release();
    // compute gradient

    DataStructures::CUDAImage Lx;
    DataStructures::CUDAImage Ly;

    Lx.Allocate(smoothedGrayFloatImage.width_, smoothedGrayFloatImage.height_, smoothedGrayFloatImage.channels_, smoothedGrayFloatImage.elementType_, smoothedGrayFloatImage.pitchedAllocation_);
    Ly.Allocate(smoothedGrayFloatImage.width_, smoothedGrayFloatImage.height_, smoothedGrayFloatImage.channels_, smoothedGrayFloatImage.elementType_, smoothedGrayFloatImage.pitchedAllocation_);

    // scharr filter for Lx

    std::vector<float> scharrXHorizontalKernel = {-1.0f, 0.0f, 1.0f};
    std::vector<float> scharrXVerticalKernel = {3.0f, 10.0f, 3.0f};

    verticalKernel.CopyFromRawHostPointer(
            scharrXVerticalKernel.data(),
            scharrXVerticalKernel.size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    horizontalKernel.CopyFromRawHostPointer(
            scharrXHorizontalKernel.data(),
            scharrXHorizontalKernel.size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    checkNppErrors(nppiFilterRowBorder_32f_C1R((float*)smoothedGrayFloatImage.gpuData_, smoothedGrayFloatImage.pitch_, floatConvertionRoi, srcOffset, (float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, (float*)verticalKernel.gpuData_, verticalKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    checkNppErrors(nppiFilterColumnBorder_32f_C1R((float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, srcOffset, (float*)Lx.gpuData_, Lx.pitch_, floatConvertionRoi, (float*)horizontalKernel.gpuData_, horizontalKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    horizontalKernel.Release();
    verticalKernel.Release();

    // scharr filter for Ly

    std::vector<float> scharrYHorizontalKernel = {3.0f, 10.0f, 3.0f};
    std::vector<float> scharrYVerticalKernel = {-1.0f, 0.0f, 1.0f};

    verticalKernel.CopyFromRawHostPointer(
            scharrYVerticalKernel.data(),
            scharrYVerticalKernel.size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    horizontalKernel.CopyFromRawHostPointer(
            scharrYHorizontalKernel.data(),
            scharrYHorizontalKernel.size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    checkNppErrors(nppiFilterRowBorder_32f_C1R((float*)smoothedGrayFloatImage.gpuData_, smoothedGrayFloatImage.pitch_, floatConvertionRoi, srcOffset, (float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, (float*)verticalKernel.gpuData_, verticalKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    checkNppErrors(nppiFilterColumnBorder_32f_C1R((float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, srcOffset, (float*)Ly.gpuData_, Ly.pitch_, floatConvertionRoi, (float*)horizontalKernel.gpuData_, horizontalKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    horizontalKernel.Release();
    verticalKernel.Release();

    compute_gradients_api((float*)Lx.gpuData_, (float*)Ly.gpuData_, (float*)tempImage.gpuData_, tempImage.width_, tempImage.height_, tempImage.pitch_, tempImage.channels_, cudaStream);
    checkCudaErrors(cudaStreamSynchronize(cudaStream));
    checkCudaErrors(cudaDeviceSynchronize());
    //find max gradient
    int scratchBufferSize = 0;
    float* maxGradientPtr = nullptr;
    checkCudaErrors(cudaMalloc(&maxGradientPtr, sizeof(float)));
    checkNppErrors(nppiMaxGetBufferHostSize_32f_C1R(floatConvertionRoi, &scratchBufferSize));
    DataStructures::CUDAImage scratchBuffer;
    scratchBuffer.Allocate(scratchBufferSize, 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, false);
    checkNppErrors(nppiMax_32f_C1R((float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, scratchBuffer.gpuData_, maxGradientPtr));
    float maxGradient = 0;
    checkCudaErrors(cudaMemcpy(&maxGradient, maxGradientPtr, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(maxGradientPtr));
    scratchBuffer.Release();

    // compute histogramm
    const size_t nbins = 300;
    scratchBufferSize = 0;
    std::vector<int> histogram(nbins + 1);
    std::vector<float> levels;
    for(int i = 0; i < nbins + 2; ++i)
    {
        levels.push_back(i);
    }
    DataStructures::CUDAImage histogramGpu;
    DataStructures::CUDAImage levelsGpu;
    histogramGpu.Allocate(nbins + 1, 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32S, false);
    levelsGpu.CopyFromRawHostPointer(levels.data(), levels.size(), 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, false);
    preprocess_histogram_api((float*)tempImage.gpuData_, tempImage.width_, tempImage.height_, tempImage.pitch_, nbins, maxGradient, cudaStream);
    checkCudaErrors(cudaStreamSynchronize(cudaStream));
    checkNppErrors(nppiHistogramRangeGetBufferSize_32f_C1R(floatConvertionRoi, nbins + 1, &scratchBufferSize));
    scratchBuffer.Allocate(scratchBufferSize, 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, false);
    checkNppErrors(nppiHistogramRange_32f_C1R((float*)tempImage.gpuData_, tempImage.pitch_, floatConvertionRoi, (Npp32s*)histogramGpu.gpuData_, (float*)levelsGpu.gpuData_, nbins + 1, scratchBuffer.gpuData_));
    histogramGpu.CopyToRawHostPointer(histogram.data(), nbins + 1, 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32S);

    size_t
    /// Compute nonlinear scale space



    /// Finalize

    checkCudaErrors(cudaStreamSynchronize(cudaStream));


    checkCudaErrors(cudaStreamDestroy(cudaStream));
    return 0;
}
