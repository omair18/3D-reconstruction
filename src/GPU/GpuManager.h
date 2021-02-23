#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <vector>
#include <memory>

#include "GPU.h"

namespace Config
{
    class JsonConfig;
}

namespace GPU
{

struct GPU;

class GpuManager
{
public:

    void UpdateCUDACapableDevicesList();

    unsigned int GetCUDACapableDevicesAmount();

    const std::vector<GPU>& GetCUDACapableDevicesList();

    GPU& SelectMatchingGPU(const std::shared_ptr<Config::JsonConfig>& config);

    void SetDevice(GPU& gpu);

private:

    std::vector<GPU> cudaCapableDevices_;
};

}

#endif // GPU_MANAGER_H
