#include <vector>
#include <iostream>

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

int main()
{
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

    return 0;
}

