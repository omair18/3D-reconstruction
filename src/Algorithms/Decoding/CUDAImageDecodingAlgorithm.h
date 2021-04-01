#ifndef IMAGE_DECODING_ALGORITHM_H
#define IMAGE_DECODING_ALGORITHM_H

#include <vector>

#include "IGPUAlgorithm.h"

namespace Decoding
{
    class IImageDecoder;
}

namespace Algorithms
{

class CUDAImageDecodingAlgorithm : public IGPUAlgorithm
{

public:

    CUDAImageDecodingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                            [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                            [[maybe_unused]] void* cudaStream);

    ~CUDAImageDecodingAlgorithm() override;

    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    std::vector<std::shared_ptr<Decoding::IImageDecoder>> decoders_;


};

}

#endif // IMAGE_DECODING_ALGORITHM_H
