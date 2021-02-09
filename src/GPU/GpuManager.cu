#include <cuda_runtime_api.h>
#include "GpuManager.h"

unsigned int GpuManager::GetCUDACapableDevicesAmount()
{
    int devicesAmount;
    cudaGetDeviceCount(&devicesAmount);
    return devicesAmount;
}

int GpuManager::GetDeviceMemoryTotal(int deviceID)
{
    cudaDeviceProp deviceProperties{};
    cudaGetDeviceProperties(&deviceProperties, deviceID);

    return deviceProperties.totalGlobalMem;
}

int GpuManager::GetDeviceMemoryFree(int deviceID)
{
    cudaDeviceProp deviceProperties{};
    cudaGetDeviceProperties(&deviceProperties, deviceID);

    return deviceProperties.res;
}

int GpuManager::GetMostFreeDevice()
{
    return 0;
}

double GpuManager::GetDeviceMemoryUsagePercent(int deviceID)
{
    return 0;
}

void GpuManager::SetDevice(int deviceID)
{

}
