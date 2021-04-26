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
    std::vector<float> Lx(3024*4032, 3);
    std::vector<float> Ly(3024*4032, 4);
    std::vector<float> result(3024*4032, 0);

    for(unsigned int i = 0; i < 3024*4032; ++i)
    {
        compute_gradients_kernel(i, Lx.data(), Ly.data(), result.data(), )
    }

    return 0;
}

