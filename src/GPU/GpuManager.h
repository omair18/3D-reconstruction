#ifndef GPUMANAGER_H
#define GPUMANAGER_H

class GpuManager
{
public:
    static unsigned int GetCUDACapableDevicesAmount();
    static int GetMostFreeDevice();
    static double GetDeviceMemoryUsagePercent(int deviceID);
    static void SetDevice(int deviceID);
};


#endif //GPUMANAGER_H
