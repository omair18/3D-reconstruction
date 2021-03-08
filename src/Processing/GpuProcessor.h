#ifndef GPU_PROCESSOR_H
#define GPU_PROCESSOR_H

#include <driver_types.h>

#include "IProcessor.h"

namespace Processing
{

class GpuProcessor final : public IProcessor
{

public:

    explicit GpuProcessor(const std::shared_ptr<Config::JsonConfig>& config);

    void Process() override;

    ~GpuProcessor() override;

private:
    cudaStream_t cudaStream_ = nullptr;

};

}


#endif // GPU_PROCESSOR_H