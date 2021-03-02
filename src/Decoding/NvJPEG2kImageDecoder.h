#ifndef NVJPEG2K_DECODER_H
#define NVJPEG2K_DECODER_H

#include <nvjpeg2k.h>
#include <npp.h>

#include "IImageDecoder.h"

namespace Decoding
{

class NvJPEG2kImageDecoder final : public IImageDecoder
{
public:

    NvJPEG2kImageDecoder();

    ~NvJPEG2kImageDecoder() override;

    void Decode(const unsigned char *data, unsigned long long size, cv::Mat &decodedData) override;

    void Decode(const unsigned char *data, unsigned long long size, cv::cuda::GpuMat &decodedData) override;

    bool IsInitialized() override;


private:
    void DecodeInternal(const unsigned char* data, unsigned long long size, cv::cuda::GpuMat& outputImage);

    void AllocateBuffer(int width, int height, int channels);

    void InitDecoder();

    cudaStream_t cudaStream_;
};

}

#endif // NVJPEG2K_DECODER_H
