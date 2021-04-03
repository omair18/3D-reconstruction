/**
 * @file GPU.h
 *
 * @brief Declares the GPU structure. This struct is used to describe GPU.
 */

#ifndef GPU_H
#define GPU_H

#include <string>

/**
 * @namespace GPU
 *
 * @brief Namespace of libgpu library.
 */
namespace GPU
{

/**
 * @struct GPU
 *
 * @brief Struct for GPU description.
 */
struct GPU
{
    /// Id of GPU.
    int deviceId_;

    /// Name of CUDA-capable device.
    std::string name_;

    /// Major version of compute capability.
    int computeCapabilityMajor_;

    /// Minor version of compute capability.
    int computeCapabilityMinor_;

    /// Amount of SMs on GPU.
    int multiprocessorsAmount_;

    /// Global memory available on device in bytes.
    std::size_t memoryTotal_;

    /// Bandwidth of GPU memory in GB/s.
    double memoryBandwidth_;

    /// Maximum resident threads per multiprocessor.
    int maxThreadsPerMultiprocessor_;

    ///  Maximum number of threads per block.
    int maxThreadsPerBlock_;

    /// Shared memory available per block in bytes.
    std::size_t sharedMemPerBlock_;

};

}

#endif // GPU_H
