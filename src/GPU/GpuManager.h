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
     * @brief Finds all CUDA-capable devices and sets information about them to the cudaCapableDevices_
     */
    void UpdateCUDACapableDevicesList();

    /**
     * @brief Returns amount of CUDA-capable devices in case of success. In other case returns 0 and puts a record
     * in a log file with ERROR severity.
     *
     * @return Amount of CUDA-capable devices.
     */
    unsigned int GetCUDACapableDevicesAmount();

    /**
     * @brief Returns CUDA-capable devices list.
     *
     * @return CUDA-capable devices list.
     */
    const std::vector<GPU>& GetCUDACapableDevicesList();

    /**
     * @brief Selects GPU from all CUDA-capable devices according to GPU selection policy.
     *
     * @param serviceConfig - service configuration with GPU selection policy
     * @return L-value reference to matching GPU.
     */
    GPU& SelectMatchingGPU(const std::shared_ptr<Config::JsonConfig>& serviceConfig);

    /**
     * @brief Selects gpu-param as current GPU for all service's processors.
     *
     * @param gpu - CUDA-capable device to be used
     */
    void SetDevice(GPU& gpu);

    /**
     * @brief Provides a constant shared pointer to a current CUDA-capable device selected by SetDevice method.
     *
     * @return Constant shared pointer to a current CUDA-capable device selected by SetDevice method.
     */
    const std::shared_ptr<GPU>& GetCurrentGPU();

private:

    /// List of CUDA-capable devices
    std::vector<GPU> cudaCapableDevices_;

    /// A pointer to a GPU selected by method SetDevice
    std::shared_ptr<GPU> selectedGPU_;
};

}

#endif // GPU_MANAGER_H
