#include <cstdio>
#include <cuda_runtime.h>

__global__ void print_elements_kernel(unsigned char* ptr, int elementsCount)
{
    for(int i = 0; i < elementsCount; i++)
        printf("%f ", ((float*)ptr)[i]);
    printf("\n\0");
}

void print_elements_api(int blocks, int threads, int elementsCount, unsigned char* devPtr)
{
    print_elements_kernel<<<blocks, threads>>>(devPtr, elementsCount);
}

__global__ void compute_gradients_kernel(float* dataLx, float* dataLy, float* resultData, size_t width, size_t pitch, size_t channels, unsigned int taskSize)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
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

void compute_gradients_api(float* dataLx, float* dataLy, float* resultData, size_t width, size_t height, size_t pitch, size_t channels, void* cudaStream)
{
    // 1 thread - 1 element
    // 1 block - 512 threads
    const unsigned int taskSize = width * height * channels;

    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    compute_gradients_kernel<<<gridSize, blockSize, 0, (cudaStream_t)cudaStream>>>(dataLx, dataLy, resultData, width, pitch / sizeof(float), channels, taskSize);
}

__global__ void preprocess_histogram_kernel(float* data, size_t width, size_t pitch, int nbins, float maxGradient, unsigned int taskSize)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / width ;
    const unsigned int elementColumn = threadId - (elementRow * width);
    const auto fnbins = (float)nbins;
    if(threadId < taskSize)
    {
        float value = data[elementRow * pitch + elementColumn];
        float processedValue = value >= 0 ? (std::floor((value / maxGradient) * fnbins)) : fnbins + 1;

        if (processedValue == fnbins)
        {
            processedValue -= 1;
        }

        if (processedValue == fnbins + 1)
        {
            processedValue = fnbins;
        }
        data[elementRow * pitch + elementColumn] = processedValue;
    }
}

void preprocess_histogram_api(float* data, size_t width, size_t height, size_t pitch, int nbins, float maxGradient, void* cudaStream)
{
    // 1 thread - 1 element
    // 1 block - 512 threads
    const unsigned int taskSize = width * height;

    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    preprocess_histogram_kernel<<<gridSize, blockSize, 0, (cudaStream_t)cudaStream>>>(data, width, pitch / sizeof(float), nbins, maxGradient, taskSize);
}

__global__ void linear_sampling_kernel(float* in, float* out, size_t in_width, size_t in_height, size_t in_pitch, size_t out_width, size_t out_height, size_t out_pitch, unsigned int taskSize)
{
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / out_width ;
    const unsigned int elementColumn = threadId - (elementRow * out_width);

    if (threadId < taskSize)
    {
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
}

void linear_sampling_api(float* in, float* out, size_t in_width, size_t in_height, size_t in_pitch, size_t out_width, size_t out_height, size_t out_pitch)
{
    // 1 thread - 1 element
    // 1 block - 512 threads
    const unsigned int taskSize = out_width * out_height;
    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    linear_sampling_kernel<<<gridSize, blockSize>>>(in, out, in_width, in_height, in_pitch / sizeof(float), out_width, out_height, out_pitch / sizeof(float), taskSize);
}

__global__ void perona_malik_g1_diffusion_kernel(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastCoefficient, float* out_data, size_t out_width, size_t out_height, size_t out_pitch, unsigned int taskSize)
{
    // e ^ -((Lx^2 + Ly^2) / K^2)
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / (out_width);
    const unsigned int elementColumn = threadId - (elementRow * out_width);

    if (threadId < taskSize)
    {
        float Lx = Lx_data[elementRow * Lx_pitch + elementColumn];
        float Ly = Ly_data[elementRow * Ly_pitch + elementColumn];

        float diffusionCoefficient = expf( (Lx * Lx + Ly * Ly) * contrastCoefficient);

        out_data[elementRow * out_pitch + elementColumn] = diffusionCoefficient;
    }
}

void perona_malik_g1_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch)
{
    const unsigned int taskSize = out_width * out_height;
    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    perona_malik_g1_diffusion_kernel<<<gridSize, blockSize>>>(Lx_data, Lx_pitch / sizeof(float), Ly_data, Ly_pitch / sizeof(float), -1 / (contrastFactor * contrastFactor), out_data, out_width, out_height, out_pitch / sizeof(float), taskSize);

}

__global__ void perona_malik_g2_diffusion_kernel(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastCoefficient, float* out_data, size_t out_width, size_t out_height, size_t out_pitch, unsigned int taskSize)
{
    // 1 / (1 + (Lx^2 + Ly^2) / K^2)
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / (out_width);
    const unsigned int elementColumn = threadId - (elementRow * out_width);

    if (threadId < taskSize)
    {
        float Lx = Lx_data[elementRow * Lx_pitch + elementColumn];
        float Ly = Ly_data[elementRow * Ly_pitch + elementColumn];

        float diffusionCoefficient = 1.f / (1 + (Lx * Lx + Ly * Ly) * contrastCoefficient);

        out_data[elementRow * out_pitch + elementColumn] = diffusionCoefficient;
    }
}

void perona_malik_g2_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch)
{
    const unsigned int taskSize = out_width * out_height;
    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    perona_malik_g2_diffusion_kernel<<<gridSize, blockSize>>>(Lx_data, Lx_pitch / sizeof(float), Ly_data, Ly_pitch / sizeof(float), 1 / (contrastFactor * contrastFactor), out_data, out_width, out_height, out_pitch / sizeof(float), taskSize);

}

__global__ void weickert_diffusion_kernel(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastCoefficient, float* out_data, size_t out_width, size_t out_height, size_t out_pitch, unsigned int taskSize)
{
    // 1 - e^(-3.315 / ((Lx^2 + Ly^2) / K^2))
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / (out_width);
    const unsigned int elementColumn = threadId - (elementRow * out_width);

    if (threadId < taskSize)
    {
        float Lx = Lx_data[elementRow * Lx_pitch + elementColumn];
        float Ly = Ly_data[elementRow * Ly_pitch + elementColumn];

        float expDivider = (Lx * Lx + Ly * Ly) * contrastCoefficient;
        float expDivider_pow2 = expDivider * expDivider;
        float expDivider_pow4 = expDivider_pow2 * expDivider_pow2;

        float diffusionCoefficient = 1.f - expf(-3.315f / expDivider_pow4);

        out_data[elementRow * out_pitch + elementColumn] = diffusionCoefficient;
    }
}

void weickert_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch)
{
    const unsigned int taskSize = out_width * out_height;
    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    weickert_diffusion_kernel<<<gridSize, blockSize>>>(Lx_data, Lx_pitch / sizeof(float), Ly_data, Ly_pitch / sizeof(float), 1 / (contrastFactor * contrastFactor), out_data, out_width, out_height, out_pitch / sizeof(float), taskSize);
}

__global__ void charbonier_diffusion_kernel(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastCoefficient, float* out_data, size_t out_width, size_t out_height, size_t out_pitch, unsigned int taskSize)
{
    // 1 / sqrt(1 + (Lx^2 + Ly^2) / K^2)
    const unsigned int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int elementRow = threadId / (out_width);
    const unsigned int elementColumn = threadId - (elementRow * out_width);

    if (threadId < taskSize)
    {
        float Lx = Lx_data[elementRow * Lx_pitch + elementColumn];
        float Ly = Ly_data[elementRow * Ly_pitch + elementColumn];

        float diffusionCoefficient = 1.f / sqrtf(1.f + (Lx * Lx + Ly * Ly) * contrastCoefficient);

        out_data[elementRow * out_pitch + elementColumn] = diffusionCoefficient;
    }
}

void charbonier_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch)
{
    const unsigned int taskSize = out_width * out_height;
    dim3 blockSize(512);
    dim3 gridSize ((taskSize + 512 - 1) / (512));

    charbonier_diffusion_kernel<<<gridSize, blockSize>>>(Lx_data, Lx_pitch / sizeof(float), Ly_data, Ly_pitch / sizeof(float), 1 / (contrastFactor * contrastFactor), out_data, out_width, out_height, out_pitch / sizeof(float), taskSize);

}