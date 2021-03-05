#ifndef GPU_H
#define GPU_H

#include <string>

/**
 * @namespace
 *
 * @brief
 */
namespace GPU
{

/**
 * @struct GPU
 *
 * @brief
 */
struct GPU
{
    ///
    unsigned int deviceId_;

    ///
    std::string name_;

    ///
    unsigned int computeCapabilityMajor_;

    ///
    unsigned int computeCapabilityMinor_;

    ///
    unsigned int multiprocessorsAmount_;

    ///
    std::size_t memoryTotal_;

    ///
    double memoryBandwidth_;
};

}

#endif // GPU_H
