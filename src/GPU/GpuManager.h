/**
 * @file GpuManager.h
 *
 * @brief Declares the GpuManager class. This class is used for providing GPU to service.
 */

#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H

#include <vector>
#include <memory>

#include "GPU.h"

// forward declaration for Config::JsonConfig
namespace Config
{
    class JsonConfig;
}

/**
 * @namespace GPU
 *
 * @brief Namespace of libgpu library.
 */
namespace GPU
{

// forward declaration for GPU::GPU
struct GPU;

/**
 * @class GpuManager
 *
 * @brief A class for providing GPU to service.
 */
class GpuManager
{
public:

    /**
     * @brief
     */
    void UpdateCUDACapableDevicesList();

    /**
     * @brief
     *
     * @return
     */
    unsigned int GetCUDACapableDevicesAmount();

    /**
     * @brief
     *
     * @return
     */
    const std::vector<GPU>& GetCUDACapableDevicesList();

    /**
     * @brief
     *
     * @param config
     * @return
     */
    GPU& SelectMatchingGPU(const std::shared_ptr<Config::JsonConfig>& config);

    /**
     * @brief Selects gpu-param as current GPU for all service's processors.
     *
     * @param gpu - CUDA-capable device to be used.
     */
    void SetDevice(GPU& gpu);

private:

    /// List of CUDA-capable devices
    std::vector<GPU> cudaCapableDevices_;
};

}

#endif // GPU_MANAGER_H
