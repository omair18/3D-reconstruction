#ifndef IMAGE_DECODING_ALGORITHM_H
#define IMAGE_DECODING_ALGORITHM_H

#include <vector>

#include "ICPUAlgorithm.h"

namespace Decoding
{
    class IImageDecoder;
}

namespace Algorithms
{

class ImageDecodingAlgorithm : public ICPUAlgorithm
{

public:

    ImageDecodingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config,
                               [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager,
                               [[maybe_unused]] void* cudaStream);

    ~ImageDecodingAlgorithm() override;

    bool Process(std::shared_ptr<DataStructures::ProcessingData> &processingData) override;

    void Initialize(const std::shared_ptr<Config::JsonConfig>& config) override;

private:

    std::vector<std::shared_ptr<Decoding::IImageDecoder>> decoders_;

};

}

#endif // IMAGE_DECODING_ALGORITHM_H
