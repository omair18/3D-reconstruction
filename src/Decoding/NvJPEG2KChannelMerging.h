#ifndef NVJPEG2K_CHANNEL_MERGING_H
#define NVJPEG2K_CHANNEL_MERGING_H

#include <cstddef>

namespace Decoding
{
    /**
     * @brief CUDA kernel API for merging NvJPEG2K buffer channels into one image.
     *
     * @param width - Width of the image
     * @param height - Height of the image
     * @param channels - Amount of channels
     * @param elementSize - Size of single element of channel buffer in bits
     * @param outputPitch - Pitch of the output image
     * @param channelBuffers - Host pointer to linear array of device pointers to channels data
     * @param channelPitches - Host pointer to linear array of channel pitches
     * @param output - Device pointer to output image data
     * @param cudaStream - CUDA stream where merging will be performed
     */
    void MergeChannels(int width, int height, int channels, int elementSize, int outputPitch, void** channelBuffers, size_t* channelPitches, void* output, void* cudaStream);
}

#endif // NVJPEG2K_CHANNEL_MERGING_H
