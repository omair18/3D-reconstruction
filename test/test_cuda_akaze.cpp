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

void show_cuda_image(DataStructures::CUDAImage& image, const std::string& windowName)
{
    cv::Mat hostImage;
    image.CopyToCvMat(hostImage);
    cv::imshow(windowName, hostImage);
    cv::waitKey(0);
}

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

struct Params
{
    float fThreshold = 0.0008f;  ///< Hessian determinant threshold
    float fDesc_factor = 1.f;   ///< Magnifier used to describe an interest point
};

struct AKAZEOptions
{
    int octaves = 4;                                          ///< Octave to process
    int omin;                                                 ///< Initial octave level (-1 means that the size of the input image is duplicated)
    int omax = 4;                                             ///< Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    int nsublevels = 4;                                       ///< Default number of sublevels per scale level
    int img_width;                                            ///< Width of the input image
    int img_height;                                           ///< Height of the input image
    float sigma0 = 1.6f;                                     ///< Base scale offset (sigma units) (used to suppress low level noise)
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

/*
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
*/
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

/**
* Compute if a number is prime of not
* @param i Input number to test
* @retval true if number is prime
* @retval false if number is not prime
*/
inline bool IsPrime( const int i )
{
    if (i == 1)
    {
        return false;
    }
    if (i == 2 || i == 3)
    {
        return true;
    }
    if (i % 2 == 0)
    {
        return false;
    }

    const size_t i_root = static_cast<int>( sqrt( static_cast<double>( i + 1 ) ) );

    for (size_t cur = 3; cur <= i_root; cur += 2)
    {
        if (i % cur == 0)
        {
            return false;
        }
    }
    return true;
}

/**
* @brief Get the next prime number greater or equal to input
* @param i Input number
* @return next prime greater or equal to input
*/
inline int NextPrimeGreaterOrEqualTo( const int i )
{
    if (IsPrime( i ))
    {
        return i;
    }
    else
    {
        int cur = i + 1;

        while (!IsPrime( cur ))
        {
            ++cur;
        }
        return cur;
    }
}

/**
 ** Compute FED cycle timings using total time
 ** @param T total time
 ** @param Tmax cycle stability limit (max : 0.25)
 ** @param tau vector of FED cycle timings
 ** @return number of cycle timings
 **/
int FEDCycleTimings(const float T, const float Tmax, std::vector<float>& tau)
{
    // Number of timings
    const int n = ceilf( sqrtf( (3.0f * T) / Tmax + 0.25f ) - 0.5f - 1.0e-8f) + 0.5f;

    // Scaling factor
    const float scale = 3.0f * T / ( Tmax * static_cast<float>( n * (n + 1)) );

    // only constants
    const float cos_fact = 1.0f / ( static_cast<float>( 4 * n ) + 2.0f );
    const float glo_fact = scale * Tmax / 2.0f;

    // Compute cycle timings
    tau.resize( n );
    for (int j = 0; j < n; ++j)
    {
        const float cos_j = cosf( M_PI * ( static_cast<float>( 2 * j + 1 ) ) * cos_fact );
        tau[ j ] = glo_fact / ( cos_j * cos_j );
    }

    // Compute Kappa reordering using kappa = n / 2
    const int kappa = n / 2;

    const int p = NextPrimeGreaterOrEqualTo( n + 1 );

    // Store new positions
    std::vector<float> tmp( n );
    for (int i = 0 , k = 0; i < n; ++i , ++k)
    {
        // Search new index
        int index = n;
        while (( index = ( ( k + 1 ) * kappa ) % p - 1 ) >= n)
        {
            ++k;
        }

        tmp[ i ] = tau[ index ];
    }

    // Get new vector
    std::swap( tmp , tau );
    return n;
}

/**
** Apply Fast Explicit Diffusion to an Image (on central part)
** @param src input image
** @param diff diffusion coefficient image
** @param half_t Half diffusion time
** @param out Output image
** @param row_start Row range beginning (range is [row_start; row_end [ )
** @param row_end Row range end (range is [row_start; row_end [ )
**/
void ImageFEDCentral( const DataStructures::CUDAImage& src , const DataStructures::CUDAImage& diff , const float half_t , DataStructures::CUDAImage& out ,
                      const int row_start , const int row_end )
{
    const int width = src.Width();
    float n_diff[4];
    float n_src[4];
    // Compute FED step on general range
    for (int i = row_start; i < row_end; ++i)
    {
        for (int j = 1; j < width - 1; ++j)
        {
            // Retrieve neighbors : TODO check if we need a cache efficient version ?
            n_diff[0] = diff( i , j + 1 );
            n_diff[1] = diff( i - 1 , j );
            n_diff[2] = diff( i , j - 1 );
            n_diff[3] = diff( i + 1 , j );
            n_src[0] = src( i , j + 1 );
            n_src[1] = src( i - 1 , j );
            n_src[2] = src( i , j - 1 );
            n_src[3] = src( i + 1 , j );

            // Compute diffusion factor for given pixel
            const float cur_src = src( i , j );
            const float cur_diff = diff( i , j );
            const float a = ( cur_diff + n_diff[0] ) * ( n_src[0] - cur_src );
            const float b = ( cur_diff + n_diff[1] ) * ( cur_src - n_src[1] );
            const float c = ( cur_diff + n_diff[2] ) * ( cur_src - n_src[2] );
            const float d = ( cur_diff + n_diff[3] ) * ( n_src[3] - cur_src );
            const float value = half_t * ( a - c + d - b );
            out( i , j ) = value;
        }
    }
}

/**
** Apply Fast Explicit Diffusion to an Image (on central part)
** @param src input image
** @param diff diffusion coefficient image
** @param half_t Half diffusion time
** @param out Output image
**/
void ImageFEDCentralCPPThread(const DataStructures::CUDAImage& src , const DataStructures::CUDAImage& diff , const float half_t , DataStructures::CUDAImage & out )
{
    const int nb_thread = omp_get_max_threads();

    // Compute ranges
    std::vector<int > range;
    SplitRange( 1 , ( int ) ( src.rows() - 1 ) , nb_thread , range );

#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < static_cast<int>( range.size() ); ++i)
    {
        ImageFEDCentral( src, diff, half_t, out, range[i - 1] , range[i] );
    }
}

/**
** Apply Fast Explicit Diffusion of an Image
** @param src input image
** @param diff diffusion coefficient image
** @param t diffusion time
** @param out output image
**/
void ImageFED(const DataStructures::CUDAImage& src, const DataStructures::CUDAImage& diff, const float tt, DataStructures::CUDAImage& out )
{
    const int width = src.Width();
    const int height = src.Height();
    const float half_t = t * 0.5f ;
    if (out.Width() != width || out.Height() != height)
    {
        out.resize( width , height );
    }
    float n_diff[4];
    float n_src[4];

    // Take care of the central part
    ImageFEDCentralCPPThread( src , diff , half_t , out );

    // Take care of the border
    // - first/last row
    // - first/last col

    // Compute FED step on first row
    for (int j = 1; j < width - 1; ++j)
    {
        n_diff[0] = diff( 0 , j + 1 );
        n_diff[2] = diff( 0 , j - 1 );
        n_diff[3] = diff( 1 , j );
        n_src[0] = src( 0 , j + 1 );
        n_src[2] = src( 0 , j - 1 );
        n_src[3] = src( 1 , j );

        // Compute diffusion factor for given pixel
        const float cur_src = src( 0 , j );
        const float cur_diff = diff( 0 , j );
        const float a = ( cur_diff + n_diff[0] ) * ( n_src[0] - cur_src );
        const float c = ( cur_diff + n_diff[2] ) * ( cur_src - n_src[2] );
        const float d = ( cur_diff + n_diff[3] ) * ( n_src[3] - cur_src );
        const float value = half_t * ( a - c + d );
        out( 0 , j ) = value;
    }

    // Compute FED step on last row
    for (int j = 1; j < width - 1; ++j)
    {
        n_diff[0] = diff( height - 1 , j + 1 );
        n_diff[1] = diff( height - 2 , j );
        n_diff[2] = diff( height - 1 , j - 1 );
        n_src[0] = src( height - 1 , j + 1 );
        n_src[1] = src( height - 2 , j );
        n_src[2] = src( height - 1 , j - 1 );

        // Compute diffusion factor for given pixel
        const float cur_src = src( height - 1 , j );
        const float cur_diff = diff( height - 1 , j );
        const float a = ( cur_diff + n_diff[0] ) * ( n_src[0] - cur_src );
        const float b = ( cur_diff + n_diff[1] ) * ( cur_src - n_src[1] );
        const float c = ( cur_diff + n_diff[2] ) * ( cur_src - n_src[2] );
        const float value = half_t * ( a - c - b );
        out( height - 1 , j ) = value;
    }

    // Compute FED step on first col
    for (int i = 1; i < height - 1; ++i)
    {
        n_diff[0] = diff( i , 1 );
        n_diff[1] = diff( i - 1 , 0 );
        n_diff[3] = diff( i + 1 , 0 );
        n_src[0] = src( i , 1 );
        n_src[1] = src( i - 1 , 0 );
        n_src[3] = src( i + 1 , 0 );

        // Compute diffusion factor for given pixel
        const float cur_src = src( i , 0 );
        const float cur_diff = diff( i , 0 );
        const float a = ( cur_diff + n_diff[0] ) * ( n_src[0] - cur_src );
        const float b = ( cur_diff + n_diff[1] ) * ( cur_src - n_src[1] );
        const float d = ( cur_diff + n_diff[3] ) * ( n_src[3] - cur_src );
        const float value = half_t * ( a + d - b );
        out( i , 0 ) = value;
    }

    // Compute FED step on last col
    for (int i = 1; i < height - 1; ++i)
    {
        n_diff[1] = diff( i - 1 , width - 1 );
        n_diff[2] = diff( i , width - 2 );
        n_diff[3] = diff( i + 1 , width - 1 );
        n_src[1] = src( i - 1 , width - 1 );
        n_src[2] = src( i , width - 2 );
        n_src[3] = src( i + 1 , width - 1 );

        // Compute diffusion factor for given pixel
        const float cur_src = src( i , width - 1 );
        const float cur_diff = diff( i , width - 1 );
        const float b = ( cur_diff + n_diff[1] ) * ( cur_src - n_src[1] );
        const float c = ( cur_diff + n_diff[2] ) * ( cur_src - n_src[2] );
        const float d = ( cur_diff + n_diff[3] ) * ( n_src[3] - cur_src );
        const float value = half_t * ( - c + d - b );
        out( i , width - 1 ) = value;
    }
}

/**
 ** Compute Fast Explicit Diffusion cycle
 ** @param self input/output image
 ** @param diff diffusion coefficient
 ** @param tau cycle timing vector
 **/
void ImageFEDCycle(DataStructures::CUDAImage& in, const DataStructures::CUDAImage& diff, DataStructures::CUDAImage& temp, const std::vector<float>& tau )
{
    for (int i = 0; i < tau.size(); ++i)
    {
        fast_explicit_diffusion_api((float*)in.gpuData_, (float*)diff.gpuData_, (float*)temp.gpuData_,
                                    in.width_, in.height_, in.pitch_, diff.pitch_, temp.pitch_);
        ImageFED( self , diff , tau[i] , temp );
        self.array() += tmp.array();
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

// Compute slice scale
static inline float ComputeSigma( const float sigma0 , const int p , const int q , const int Q )
{
    if (p == 0 && q == 0 )
        return sigma0;
    else
        return sigma0 * powf( 2.f , p + static_cast<float>( q ) / static_cast<float>( Q ) );
}

void ImageHalfSample(const DataStructures::CUDAImage& in, DataStructures::CUDAImage& out)
{
    const size_t outWidth = in.width_ / 2;
    const size_t outHeight = in.height_ / 2;

    if (!out.Allocated())
    {
        out.Allocate(outWidth, outHeight, in.channels_, in.elementType_, in.pitchedAllocation_);
    }
    else
    {
        if (out.allocatedBytes_ < (outWidth * outHeight * in.channels_ * in.GetElementSize()))
        {
            out.Allocate(outWidth, outHeight, in.channels_, in.elementType_, in.pitchedAllocation_);
        }
    }

    linear_sampling_api((float*)in.gpuData_, (float*)out.gpuData_, in.width_, in.height_, in.pitch_, outWidth, outHeight, out.pitch_);

}

void PeronaMalikG1Diffusion(DataStructures::CUDAImage& Lx, DataStructures::CUDAImage& Ly, float contrastFactor, DataStructures::CUDAImage& out)
{

}

void PeronaMalikG2Diffusion(DataStructures::CUDAImage& Lx, DataStructures::CUDAImage& Ly, float contrastFactor, DataStructures::CUDAImage& out)
{
    perona_malik_g2_diffusion_api((float*)Lx.gpuData_, Lx.pitch_, (float*)Ly.gpuData_, Ly.pitch_, contrastFactor, (float*)out.gpuData_,
                                  out.width_, out.height_, out.pitch_);
}

void WeickertDiffusion(DataStructures::CUDAImage& Lx, DataStructures::CUDAImage& Ly, float contrastFactor, DataStructures::CUDAImage& out)
{

}

void CharbonnierDiffusion(DataStructures::CUDAImage& Lx, DataStructures::CUDAImage& Ly, float contrastFactor, DataStructures::CUDAImage& out)
{

}

void ComputeHessian()
{

}

void LoadImage(const std::string& path, DataStructures::CUDAImage& out)
{
    cv::Mat image = cv::imread(path);
    out.CopyFromCvMat(image);
}

void GrayScaleImage(const DataStructures::CUDAImage& input, DataStructures::CUDAImage& output)
{
    const std::vector<float> grayscaleCoefficients = {0.114f, 0.587f, 0.299f};

    output.Allocate(input.width_, input.height_, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, true);

    NppiSize grayScaleRoi = { .width = (int)input.width_, .height = (int)input.height_ };

    checkNppErrors(nppiColorToGray_8u_C3C1R(input.gpuData_, input.pitch_, output.gpuData_, output.pitch_, grayScaleRoi, grayscaleCoefficients.data()));
}

void ConvertToFloat(const DataStructures::CUDAImage& input, DataStructures::CUDAImage& output)
{
    NppiSize floatConvertionRoi = { .width = (int)input.width_, .height = (int)input.height_ };
    output.Allocate(input.width_, input.height_, input.channels_, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F, true);

    checkNppErrors(nppiConvert_8u32f_C1R(input.gpuData_, input.pitch_, (float*)output.gpuData_, output.pitch_, floatConvertionRoi));
}

void NormalizeImage(const DataStructures::CUDAImage& input, DataStructures::CUDAImage& output)
{
    NppiSize normalizationRoi = { .width = (int)input.width_, .height = (int)input.height_ };
    output.Allocate(input.width_, input.height_, input.channels_, input.elementType_, true);

    checkNppErrors(nppiDivC_32f_C1R((float *)input.gpuData_, input.pitch_, 255.f, (float *)output.gpuData_, output.pitch_, normalizationRoi));
}

void NormalizeImage(DataStructures::CUDAImage& input)
{
    NppiSize normalizationRoi = { .width = (int)input.width_, .height = (int)input.height_ };
    checkNppErrors(nppiDivC_32f_C1R((float *)input.gpuData_, input.pitch_, 255.f, (float *)input.gpuData_, input.pitch_, normalizationRoi));
}

DataStructures::CUDAImage verticalDeviceKernel;
DataStructures::CUDAImage horizontalDeviceKernel;

void ImageSeparableConvolution(const DataStructures::CUDAImage& input, DataStructures::CUDAImage& output, DataStructures::CUDAImage& temp, const std::vector<float>& hostHorizontalKernel, const std::vector<float>& hostVerticalKernel)
{
    NppiPoint srcOffset {.x = 0, .y = 0};
    NppiSize convolutionRoi = { .width = (int)input.width_, .height = (int)input.height_ };

    verticalDeviceKernel.CopyFromRawHostPointer(
            (void*)hostVerticalKernel.data(),
            hostVerticalKernel.size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    horizontalDeviceKernel.CopyFromRawHostPointer(
            (void*)hostHorizontalKernel.data(),
            hostHorizontalKernel.size(),
            1,
            1,
            DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32F,
            false);

    checkNppErrors(nppiFilterRowBorder_32f_C1R((float*)input.gpuData_, input.pitch_, convolutionRoi, srcOffset, (float*)temp.gpuData_, temp.pitch_, convolutionRoi, (float *)verticalDeviceKernel.gpuData_, verticalDeviceKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    checkNppErrors(nppiFilterColumnBorder_32f_C1R((float*)temp.gpuData_, temp.pitch_, convolutionRoi, srcOffset, (float*)output.gpuData_, output.pitch_, convolutionRoi, (float*)horizontalDeviceKernel.gpuData_, horizontalDeviceKernel.width_, 0, NppiBorderType::NPP_BORDER_REPLICATE));

    verticalDeviceKernel.Release();
    horizontalDeviceKernel.Release();
}

// map <(sigma, size), kernel>
std::map<std::pair<float, size_t>, std::vector<float>> gaussKernelsHorizontalCache;
std::map<std::pair<float, size_t>, std::vector<float>> gaussKernelsVerticalCache;

void ImageGaussianFilter(const DataStructures::CUDAImage& sourceImage, const float sigma, DataStructures::CUDAImage& output, DataStructures::CUDAImage& temp, const size_t kernelSizeX, const size_t kernelSizeY)
{
    auto gaussHorizontalKernelKey = std::make_pair(sigma, kernelSizeX);
    auto gaussVerticalKernelKey = std::make_pair(sigma, kernelSizeY);

    std::vector<float> horizontalGaussKernel;
    std::vector<float> verticalGaussKernel;

    if (gaussKernelsHorizontalCache.find(gaussHorizontalKernelKey) == gaussKernelsHorizontalCache.end())
    {
        ComputeGaussianKernel(horizontalGaussKernel, sigma, kernelSizeX);
        gaussKernelsHorizontalCache.insert(std::make_pair(gaussHorizontalKernelKey, horizontalGaussKernel));
    }
    else
    {
        horizontalGaussKernel = gaussKernelsHorizontalCache[gaussHorizontalKernelKey];
    }

    if (gaussKernelsVerticalCache.find(gaussVerticalKernelKey) == gaussKernelsVerticalCache.end())
    {
        ComputeGaussianKernel(verticalGaussKernel, sigma, kernelSizeY);
        gaussKernelsVerticalCache.insert(std::make_pair(gaussVerticalKernelKey, verticalGaussKernel));
    }
    else
    {
        verticalGaussKernel = gaussKernelsVerticalCache[gaussVerticalKernelKey];
    }

    ImageSeparableConvolution(sourceImage, output, temp, horizontalGaussKernel, verticalGaussKernel);
}

void ImageScharrXDerivative(const DataStructures::CUDAImage& input, DataStructures::CUDAImage& output, DataStructures::CUDAImage& temp, bool normalize)
{
    std::vector<float> scharrXHorizontalKernel = {-1.0f, 0.0f, 1.0f};
    std::vector<float> scharrXVerticalKernel = {3.0f, 10.0f, 3.0f};

    if (normalize)
    {
        for(auto& i : scharrXHorizontalKernel)
        {
            i /= 2;
        }

        for(auto& i : scharrXVerticalKernel)
        {
            i /= 16;
        }
    }

    ImageSeparableConvolution(input, output, temp, scharrXHorizontalKernel, scharrXVerticalKernel);
}

void ImageScharrYDerivative(const DataStructures::CUDAImage& input, DataStructures::CUDAImage& output, DataStructures::CUDAImage& temp, bool normalize)
{
    std::vector<float> scharrYHorizontalKernel = {3.0f, 10.0f, 3.0f};
    std::vector<float> scharrYVerticalKernel = {-1.0f, 0.0f, 1.0f};

    if (normalize)
    {
        for(auto& i : scharrYHorizontalKernel)
        {
            i /= 16;
        }

        for(auto& i : scharrYVerticalKernel)
        {
            i /= 2;
        }
    }

    ImageSeparableConvolution(input, output, temp, scharrYHorizontalKernel, scharrYVerticalKernel);
}

/**
 ** Compute X-derivative using scaled Scharr filter
 ** @param img Input image
 ** @param out Output image
 ** @param scale scale of filter (1 -> 3x3 filter; 2 -> 5x5, ...)
 ** @param bNormalize true if kernel must be normalized
 **/
void ImageScaledScharrXDerivative( const DataStructures::CUDAImage& img, DataStructures::CUDAImage& out, DataStructures::CUDAImage& temp, const int scale , const bool bNormalize = true )
{
    const int kernel_size = 3 + 2 * ( scale - 1 );

    std::vector<float> kernel_vert;
    std::vector<float> kernel_horiz;

    kernel_vert.resize(kernel_size);
    kernel_horiz.resize(kernel_size);


    /*
    General X-derivative function
                                | -1   0   1 |
    D = 1 / ( 2 h * ( w + 2 ) ) | -w   0   w |
                                | -1   0   1 |
    */

    std::fill(kernel_horiz.begin(), kernel_horiz.end(), 0);
    kernel_horiz[0] = -1.0;
    // kernel_horiz( kernel_size / 2 ) = 0.0;
    kernel_horiz[kernel_size - 1] = 1.0;

    // Scharr parameter for derivative
    const float w = 10.0f / 3.0f;

    std::fill(kernel_vert.begin(), kernel_vert.end(), 0);
    kernel_vert[0] = 1.0;
    kernel_vert[kernel_size / 2] = w;
    kernel_vert[kernel_size - 1] = 1.0;

    if (bNormalize)
    {
        for (auto& number : kernel_vert)
        {
            number *= (1.0f / (2.0f * static_cast<float>(scale) * (w + 2.0f)));
        }
    }

    ImageSeparableConvolution(img, out, temp, kernel_horiz , kernel_vert);
}



/**
 ** Compute Y-derivative using scaled Scharr filter
 ** @param img Input image
 ** @param out Output image
 ** @param scale scale of filter (1 -> 3x3 filter; 2 -> 5x5, ...)
 ** @param bNormalize true if kernel must be normalized
 **/
void ImageScaledScharrYDerivative( const DataStructures::CUDAImage& img, DataStructures::CUDAImage& out, DataStructures::CUDAImage& temp, const int scale , const bool bNormalize = true )
{
    /*
    General Y-derivative function
                                | -1  -w  -1 |
    D = 1 / ( 2 h * ( w + 2 ) ) |  0   0   0 |
                                |  1   w   1 |
    */
    const int kernel_size = 3 + 2 * ( scale - 1 );

    std::vector<float> kernel_vert;
    std::vector<float> kernel_horiz;

    kernel_vert.resize(kernel_size);
    kernel_horiz.resize(kernel_size);

    // Scharr parameter for derivative
    const float w = 10.0f / 3.0f;


    std::fill(kernel_horiz.begin(), kernel_horiz.end(), 0);
    kernel_horiz[0] = 1.0;
    kernel_horiz[kernel_size / 2] = w;
    kernel_horiz[kernel_size - 1] = 1.0;

    if (bNormalize )
    {
        for (auto& number : kernel_horiz)
        {
            number *= (1.0f / (2.0f * static_cast<float>(scale) * (w + 2.0f)));
        }
    }

    std::fill(kernel_vert.begin(), kernel_vert.end(), 0);
    kernel_vert[0] = - 1.0;
    // kernel_vert( kernel_size / 2 ) = 0.0;
    kernel_vert[kernel_size - 1] = 1.0;

    ImageSeparableConvolution(img, out, temp, kernel_horiz, kernel_vert);
}

void ComputeAKAZESlice(const DataStructures::CUDAImage& src,
                       DataStructures::CUDAImage& temp,
                       const int p,
                       const int q,
                       AKAZEOptions& options,
                       const float contrast_factor,
                       DataStructures::CUDAImage& Li, // Diffusion image
                       DataStructures::CUDAImage& Lx, // X derivatives
                       DataStructures::CUDAImage& Ly, // Y derivatives
                       DataStructures::CUDAImage& Lhess) // Det(Hessian)
{
    const float sigma_cur = ComputeSigma(options.sigma0 ,p ,q ,options.nsublevels );
    const float ratioFactor = 1 << p; //pow(2,p);
    const int sigma_scale = std::round(sigma_cur * options.derivative_factor / ratioFactor);

    DataStructures::CUDAImage smoothed;
    if (p == 0 && q == 0 )
    {
        // Compute new image

        ImageGaussianFilter(src, options.sigma0, Li, temp, 0, 0);
    }
    else
    {
        // general case
        DataStructures::CUDAImage in;
        if (q == 0 )
        {
            ImageHalfSample(src, in );
        }
        else
        {
            in = src;
        }

        const float sigma_prev = ( q == 0 ) ? ComputeSigma(options.sigma0,p - 1, options.nsublevels - 1, options.nsublevels ) : ComputeSigma(options.sigma0, p, q - 1, options.nsublevels );

        // Compute non linear timing between two consecutive slices
        const float t_prev = 0.5f * ( sigma_prev * sigma_prev );
        const float t_cur  = 0.5f * ( sigma_cur * sigma_cur );
        const float total_cycle_time = t_cur - t_prev;

        // Compute first derivatives (Scharr scale 1, non normalized) for diffusion coef
        ImageGaussianFilter(in , 1.f , smoothed, temp, 0, 0 );

        ImageScharrXDerivative(smoothed , Lx , temp, false );
        ImageScharrYDerivative(smoothed , Ly , temp, false );

        // Compute diffusion coefficient
        DataStructures::CUDAImage& diff = smoothed; // diffusivity image (reuse existing memory)
        PeronaMalikG2Diffusion(Lx, Ly, contrast_factor, diff);

        // Compute FED cycles
        std::vector<float> tau;
        FEDCycleTimings( total_cycle_time , 0.25f , tau );
        ImageFEDCycle( in , diff , tau );
        Li = in; // evolution image
    }

    // Compute Hessian response
    if (p == 0 && q == 0 )
    {
        smoothed = Li;
    }
    else
    {
        // Add a little smooth to image (for robustness of Scharr derivatives)
        ImageGaussianFilter( Li , 1.f , smoothed, temp, 0, 0 );
    }

    // Compute true first derivatives
    ImageScaledScharrXDerivative(smoothed, Lx, temp, sigma_scale );
    ImageScaledScharrYDerivative(smoothed, Ly, temp, sigma_scale );

    // Second order spatial derivatives
    Image<float> Lxx, Lyy, Lxy;
    ImageScaledScharrXDerivative( Lx , Lxx , sigma_scale );
    ImageScaledScharrYDerivative( Lx , Lxy , sigma_scale );
    ImageScaledScharrYDerivative( Ly , Lyy , sigma_scale );

    Lx *= static_cast<float>(sigma_scale);
    Ly *= static_cast<float>(sigma_scale);

    // Compute Determinant of the Hessian
    Lhess.resize(Li.Width(), Li.Height());
    const float sigma_size_square = sigma_scale * sigma_scale;
    const float sigma_size_quad = sigma_size_square * sigma_size_square;
    Lhess.array() = (Lxx.array()*Lyy.array()-Lxy.array().square())*sigma_size_quad;
}

int main()
{
    LOGGER_INIT();

    cudaStream_t cudaStream;
    checkCudaErrors(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));

    /// Prepare

    DataStructures::CUDAImage sourceImage;

    LoadImage("/home/valera/Photo/30/IMG_20201011_131521.jpg", sourceImage);

    /// Now we have BGR image on GPU.
    /// Grayscaling

    DataStructures::CUDAImage grayImage;

    GrayScaleImage(sourceImage, grayImage);

    /// Converting to float image

    DataStructures::CUDAImage grayFloatImage;

    ConvertToFloat(grayImage, grayFloatImage);

    /// Normalize float image

    DataStructures::CUDAImage grayFloatNormalizedImage;

    NormalizeImage(grayFloatImage, grayFloatNormalizedImage);

    /// Compute contrast factor

    float contrastFactor = 0;

    DataStructures::CUDAImage smoothedGrayFloatNormalizedImage;
    DataStructures::CUDAImage tempImage;

    tempImage.Allocate(grayFloatNormalizedImage.width_, grayFloatNormalizedImage.height_, grayFloatNormalizedImage.channels_, grayFloatNormalizedImage.elementType_, grayFloatNormalizedImage.pitchedAllocation_);
    smoothedGrayFloatNormalizedImage.Allocate(grayFloatNormalizedImage.width_, grayFloatNormalizedImage.height_, grayFloatNormalizedImage.channels_, grayFloatNormalizedImage.elementType_, grayFloatNormalizedImage.pitchedAllocation_);

    ImageGaussianFilter(grayFloatNormalizedImage, 1.f, smoothedGrayFloatNormalizedImage, tempImage, 0, 0);

    // compute gradient

    DataStructures::CUDAImage Lx;
    DataStructures::CUDAImage Ly;

    Lx.Allocate(smoothedGrayFloatNormalizedImage.width_, smoothedGrayFloatNormalizedImage.height_, smoothedGrayFloatNormalizedImage.channels_, smoothedGrayFloatNormalizedImage.elementType_, smoothedGrayFloatNormalizedImage.pitchedAllocation_);
    Ly.Allocate(smoothedGrayFloatNormalizedImage.width_, smoothedGrayFloatNormalizedImage.height_, smoothedGrayFloatNormalizedImage.channels_, smoothedGrayFloatNormalizedImage.elementType_, smoothedGrayFloatNormalizedImage.pitchedAllocation_);

    // scharr filter for Lx

    ImageScharrXDerivative(smoothedGrayFloatNormalizedImage, Lx, tempImage, false);

    // scharr filter for Ly

    ImageScharrYDerivative(smoothedGrayFloatNormalizedImage, Ly, tempImage, false);


    compute_gradients_api((float*)Lx.gpuData_, (float*)Ly.gpuData_, (float*)tempImage.gpuData_, tempImage.width_, tempImage.height_, tempImage.pitch_, tempImage.channels_, cudaStream);
    checkCudaErrors(cudaStreamSynchronize(cudaStream));

    //find max gradient

    NppiSize maxGradientRoi = { .width = (int)smoothedGrayFloatNormalizedImage.width_, .height = (int)smoothedGrayFloatNormalizedImage.height_ };

    int scratchBufferSize = 0;
    float* maxGradientPtr = nullptr;
    checkCudaErrors(cudaMalloc(&maxGradientPtr, sizeof(float)));
    checkNppErrors(nppiMaxGetBufferHostSize_32f_C1R(maxGradientRoi, &scratchBufferSize));
    DataStructures::CUDAImage scratchBuffer;
    scratchBuffer.Allocate(scratchBufferSize, 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, false);
    checkNppErrors(nppiMax_32f_C1R((float*)tempImage.gpuData_, tempImage.pitch_, maxGradientRoi, scratchBuffer.gpuData_, maxGradientPtr));
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
    checkNppErrors(nppiHistogramRangeGetBufferSize_32f_C1R(maxGradientRoi, nbins + 1, &scratchBufferSize));
    scratchBuffer.Allocate(scratchBufferSize, 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_8U, false);
    checkNppErrors(nppiHistogramRange_32f_C1R((float*)tempImage.gpuData_, tempImage.pitch_, maxGradientRoi, (Npp32s*)histogramGpu.gpuData_, (float*)levelsGpu.gpuData_, nbins + 1, scratchBuffer.gpuData_));
    histogramGpu.CopyToRawHostPointer(histogram.data(), nbins + 1, 1, DataStructures::CUDAImage::ELEMENT_TYPE::TYPE_32S);

    unsigned int amountOfPoints = sourceImage.width_ * sourceImage.height_;
    const float percentile = 0.7;

    int nthreshold = (int)(amountOfPoints * percentile);

    int k, nelements = 0;
    for (k = 0; nelements < nthreshold && k < nbins; k++)
    {
        nelements += histogram[k];
    }

    contrastFactor = (nelements < nthreshold ? 0.03f : maxGradient * ((float)k / nbins));

    /// Compute nonlinear scale space

    AKAZEOptions options;
    options.img_width = sourceImage.width_;
    options.img_height = sourceImage.height_;

    // Octave computation

    DataStructures::CUDAImage* input = &grayFloatNormalizedImage;

    std::vector<TEvolution> evolutions;         ///< Vector of nonlinear diffusion evolution

    auto sublevelContrastFactor = contrastFactor;

    for (int p = 0; p < options.octaves; ++p)
    {
        sublevelContrastFactor *= (p == 0 ? 1.0f : 0.75f);

        for (int q = 0; q < options.nsublevels; ++q)
        {
            TEvolution evolution;


        }
    }



    /// Finalize

    checkCudaErrors(cudaStreamSynchronize(cudaStream));
    checkCudaErrors(cudaStreamDestroy(cudaStream));
    return 0;
}
