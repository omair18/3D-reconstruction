#include <opencv2/core/mat.hpp>

#include <vector>
#include <iostream>
#include <cmath>

/*
void MergeChannels8UKernel(int id, int width, int height, int outputPitch, int channels, std::vector<std::vector<unsigned char>>& gpuArrayOfChannels, std::vector<size_t>& gpuArrayOfChannelPitches, std::vector<unsigned char>& output)
{
    const unsigned int threadId = id;
    const unsigned int elementsPerThread = 4;
    const unsigned int taskSize = width * height;
    const unsigned int elementId = threadId * elementsPerThread;
    const unsigned int elementRow = elementId / width;
    const unsigned int elementColumn = elementId - (elementRow * width);

    if(elementId < taskSize)
    {
        for(unsigned int element = 0; element < elementsPerThread; ++element)
        {
            for(unsigned int channel = 0; channel < channels; ++channel)
            {
                output[elementRow * outputPitch + elementColumn + element * channels + channel] =
                        gpuArrayOfChannels[channel][elementRow * gpuArrayOfChannelPitches[channel] + elementColumn + element];
            }
        }
    }
}
*/

/*
void elementwise_divide_float(unsigned int threadId, const float divider, float* gpuData, size_t width, size_t pitch, size_t channels, const unsigned int taskSize)
{
    const unsigned int elementRow = threadId / (width * channels);
    const unsigned int elementColumn = threadId - (elementRow * width * channels);
    if(threadId < taskSize)
    {
        gpuData[elementRow * pitch / sizeof(float)+ elementColumn] /= divider;
    }
}
*/
/*
void compute_gradients_kernel(unsigned int threadId, float* dataLx, float* dataLy, float* resultData, size_t width, size_t pitch, size_t channels, unsigned int taskSize)
{
    const unsigned int elementRow = threadId / (width * channels);
    const unsigned int elementColumn = threadId - (elementRow * width * channels);

    if(threadId < taskSize)
    {
        float dx = dataLx[elementRow * pitch + elementColumn];
        float dy = dataLy[elementRow * pitch + elementColumn];
        float grad = sqrtf(dx * dx + dy * dy);
        resultData[elementRow * pitch + elementColumn] = grad;
    }
}
*/

void linear_sampling_kernel(unsigned int threadId, std::vector<float>& in, std::vector<float>& out, size_t in_width, size_t in_height, size_t in_pitch, size_t out_width, size_t out_height, size_t out_pitch, unsigned int taskSize)
{
    const unsigned int elementRow = threadId / out_width ;
    const unsigned int elementColumn = threadId - (elementRow * out_width);
    const float y = ((float)in_height / (float)out_height) * ((float)elementRow + 0.5f);
    const float x = ((float)in_width / (float)out_width) * ((float)elementColumn + 0.5f);

    float coefs_x[2];
    float coefs_y[2];

    const float dx = x - floorf(x);
    const float dy = y - floorf(y);

    coefs_x[0] = 1.0f - dx;
    coefs_x[1] = dx;

    coefs_y[0] = 1.0f - dy;
    coefs_y[1] = dy;

    float res = 0;

    const int grid_x = static_cast<int>(floorf(x));
    const int grid_y = static_cast<int>(floorf(y));

    float total_weight = 0;

    for (int i = 0; i < 2; ++i)
    {
        const int cur_i = grid_y + i;

        if (cur_i >= 0  && cur_i < in_height)
        {
            for (int j = 0; j < 2; j++)
            {
                const int cur_j = grid_x + j;

                if (cur_j >= 0 && cur_j < in_width)
                {

                    float weight = coefs_x[j] * coefs_y[i];

                    float pixel = in[cur_i * in_pitch + cur_j];
                    float weightedPixel = pixel * weight;

                    res += weightedPixel;

                    total_weight += weight;
                }
            }
        }
    }

    float result_to_write = 0;

    if (total_weight <= 0.2)
    {
        result_to_write = 0;
    }
    else
    {
        result_to_write = res;
    }

    if (result_to_write != 1.0f)
    {
        result_to_write /= total_weight;
    }

    out[elementRow * out_pitch + elementColumn] = result_to_write;
}

template<typename T>
struct RealPixel
{
    using base_type = T;
    using real_type = double;

    /**
    * @brief Cast pixel value to real
    * @param val Input value
    * @return casted value
    */
    static real_type convert_to_real( const base_type & val )
    {
        return static_cast<real_type>( val );
    }

    /**
    * @brief Cast pixel value to base_type
    * @param val Input value
    * @return casted value
    */
    static base_type convert_from_real( const real_type & val )
    {
        return static_cast<base_type>( val );
    }
};

struct SamplerLinear
{
public:
    // Linear sampling is between two pixels
    static const int neighbor_width = 2;

    /**
     ** @brief Computes weight associated to neighboring pixels
     ** @author Romuald Perrot <perrot.romuald_AT_gmail.com>
     ** @param x Sampling position
     ** @param[out] weigth Sampling factors associated to the neighboring
     ** @note weight must be at least neighbor_width length
     **/
    void operator()( const double x , double * const weigth ) const
    {
        weigth[0] = 1.0 - x;
        weigth[1] = x;
    }
};

template<typename SamplerFunc>
struct Sampler2d
{
    explicit Sampler2d( const SamplerFunc & sampler = SamplerFunc() )
            : sampler_( sampler ) ,
              half_width_( SamplerFunc::neighbor_width / 2 )
    {

    }

    /**
     ** Sample image at a specified position
     ** @param src Input image
     ** @param y Y-coordinate of sampling
     ** @param x X-coordinate of sampling
     ** @return Sampled value
     **/
    template <typename T>
    T operator()( const cv::Mat_<T> & src , const float y , const float x ) const
    {
        const int im_width = src.cols;
        const int im_height = src.rows;

        // Get sampler coefficients
        double coefs_x[ SamplerFunc::neighbor_width ];
        double coefs_y[ SamplerFunc::neighbor_width ];

        // Compute difference between exact pixel location and sample
        const double dx = static_cast<double>( x ) - floor( x );
        const double dy = static_cast<double>( y ) - floor( y );

        // Get sampler weights
        sampler_( dx , coefs_x );
        sampler_( dy , coefs_y );

        typename RealPixel<T>::real_type res( 0 );

        // integer position of sample (x,y)
        const int grid_x = static_cast<int>( floor( x ) );
        const int grid_y = static_cast<int>( floor( y ) );

        // Sample a grid around specified grid point
        double total_weight = 0.0;
        for (int i = 0; i < SamplerFunc::neighbor_width; ++i )
        {
            // Get current i value
            // +1 for correct scheme (draw it to be conviced)
            const int cur_i = grid_y + 1 + i - half_width_;

            // handle out of range
            if (cur_i < 0 || cur_i >= im_height )
            {
                continue;
            }

            for (int j = 0; j < SamplerFunc::neighbor_width; ++j )
            {
                // Get current j value
                // +1 for the same reason
                const int cur_j = grid_x + 1 + j - half_width_;

                // handle out of range
                if (cur_j < 0 || cur_j >= im_width )
                {
                    continue;
                }


                // sample input image and weight according to sampler
                const double w = coefs_x[ j ] * coefs_y[ i ];
                const typename RealPixel<T>::real_type pix = RealPixel<T>::convert_to_real( src.template at<T>( cur_i , cur_j ) );
                const typename RealPixel<T>::real_type wp = pix * w;
                res += wp;

                total_weight += w;
            }
        }

        // If value too small, it should be so instable, so return the sampled value
        if (total_weight <= 0.2 )
        {
            return T();
        }

        if (total_weight != 1.0 )
        {
            res /= total_weight;
        }


        return RealPixel<T>::convert_from_real( res );
    }

private:

    /// Sampler function used to resample input image
    SamplerFunc sampler_;

    /// Sampling window
    const int half_width_;
};

template<typename T>
void ImageHalfSample( const cv::Mat_<T> & src , cv::Mat_<T> & out )
{
    const int new_width  = src.cols / 2;
    const int new_height = src.rows / 2;

    out.resize( new_width , new_height );

    const Sampler2d<SamplerLinear> sampler;

    for (int i = 0; i < new_height; ++i )
    {
        for (int j = 0; j < new_width; ++j )
        {
            // Use .5f offset to ensure mid pixel and correct bilinear sampling
            out.template at<T>( i , j ) =  sampler( src, 2.f * ( i + .5f ), 2.f * ( j + .5f ) );
        }
    }
}

int main()
{
    /*
    /// channle merging
    std::vector<unsigned char> c1(1280*720, 1);
    std::vector<unsigned char> c2(1280*720, 2);
    std::vector<unsigned char> c3(1280*720, 3);

    std::vector<std::vector<unsigned char>> channels;
    channels.push_back(std::move(c1));
    channels.push_back(std::move(c2));
    channels.push_back(std::move(c3));

    std::vector<size_t> pitches = {1280, 1280, 1280};

    std::vector<unsigned char> output(1280*3*720, 0);

    for(int i = 0; i < 900*256; i++)
    {
        if(i > 1283)
        {
            MergeChannels8UKernel(i, 1280, 720, 1280*3, 3,channels, pitches, output);
        }
        else
        {
            MergeChannels8UKernel(i, 1280, 720, 1280*3, 3,channels, pitches, output);
        }
    }

    for(int i = 0; i < output.size(); ++i)
    {
        if(output[i] == 0)
            std::cout << i << std::endl;
    }
    */

    /*
    /// normalization
    std::vector<float> test(3024*4032*3, 128);

    for(unsigned int i = 0; i < 3024*4032*3; ++i)
    {
        elementwise_divide_float(i, 256., test.data(), 3024, 3024*3*sizeof(float), 3, 3024*4032*3);
    }

    for(unsigned int i = 0; i < 3024*4032*3; ++i)
    {
        if (test[i] != 128/256.)
        {
            std::cout << i << std::endl;
        }
    }
*/
    /*
    std::vector<float> Lx(3024*4032, 3);
    std::vector<float> Ly(3024*4032, 4);
    std::vector<float> result(3024*4032, 0);

    for(unsigned int i = 0; i < 3024*4032; ++i)
    {
        compute_gradients_kernel(i, Lx.data(), Ly.data(), result.data(), )
    }
*/
    std::vector<float> in(3024*4032, 15);
    std::vector<float> out((3024 / 2) * (4032 / 2));

    auto out_h = (4032 / 2);
    auto out_w = (3024 / 2);

    cv::Mat_<float> a(4032, 3024, 15);
    cv::Mat_<float> b(out_h, out_w, 0.f);
    //b.resize(out_w, out_h);

    //a.fill(15);

    for (int i = 0; i < out_h; ++i)
    {
        for (int j = 0; j < out_w; ++j)
        {
            linear_sampling_kernel(i * out_w + j, in, out, 3024, 4032, 3024, out_w, out_h, out_w, out_w * out_h);
        }
    }

    ImageHalfSample(a, b);

    for (int i = 0; i < out_h; ++i)
    {
        for (int j = 0; j < out_w; ++j)
        {
            std::cout << "[ " << i << ", " << j <<"] MVG: " << b.at<float>(i, j) << "\tMy: " << out[i * out_w + j] << std::endl;
        }
    }

    return 0;
}

