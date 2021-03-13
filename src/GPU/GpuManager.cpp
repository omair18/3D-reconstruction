#include <cuda_runtime_api.h>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include "GpuManager.h"
#include "Logger.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"

namespace GPU
{

void GpuManager::UpdateCUDACapableDevicesList()
{
    LOG_TRACE() << "Updating CUDA-capable devices list...";
    cudaCapableDevices_.clear();
    unsigned int gpusCount = GetCUDACapableDevicesAmount();
    if (gpusCount < 1)
    {
        LOG_FATAL() << "At least one CUDA-capable device is required.";
        throw std::runtime_error("At least one CUDA-capable device is required.");
    }
    cudaCapableDevices_.reserve(gpusCount);
    cudaDeviceProp properties {};
    cudaError_t status;
    for (int i = 0; i < gpusCount; ++i)
    {
        status = cudaGetDeviceProperties(&properties, static_cast<int>(i));
        if(status != cudaError_t::cudaSuccess)
        {
            auto errorName = cudaGetErrorName(status);
            auto errorDescription = cudaGetErrorString(status);
            LOG_ERROR() << "Failed to get properties of device with ID " << i << ". Reason: "
                        << errorName << " - " << errorDescription;
        }
        else
        {
            GPU gpu;
            gpu.deviceId_ = i;
            gpu.computeCapabilityMajor_ = properties.major;
            gpu.computeCapabilityMinor_ = properties.minor;
            gpu.name_ = properties.name;
            gpu.multiprocessorsAmount_ = properties.multiProcessorCount;
            gpu.memoryTotal_ = properties.totalGlobalMem;
            gpu.memoryBandwidth_ = static_cast<double>(properties.memoryBusWidth) * properties.memoryClockRate / 4000000;
            gpu.maxThreadsPerBlock_ = properties.maxThreadsPerBlock;
            gpu.sharedMemPerBlock_ = properties.sharedMemPerBlock;
            gpu.maxThreadsPerMultiprocessor_ = properties.maxThreadsPerMultiProcessor;

            LOG_TRACE() << "Found GPU: [" << i << "] "
            << gpu.name_ << " with "
            << (static_cast<double>(gpu.memoryTotal_) / (1024 * 1024 * 1024)) << " GB "
            << "(" << gpu.memoryBandwidth_ << " GB/s) "
            << "SM_" << gpu.computeCapabilityMajor_ << gpu.computeCapabilityMinor_
            << " with " << gpu.multiprocessorsAmount_ << " multiprocessors.";

            cudaCapableDevices_.push_back(std::move(gpu));
        }
    }
}

void GpuManager::SetDevice(GPU& gpu)
{
    LOG_TRACE() << "Using " << gpu.name_ << " as current GPU.";
    cudaSetDevice(static_cast<int>(gpu.deviceId_));
    selectedGPU_ = std::make_shared<GPU>(gpu);
}

const std::vector<GPU>& GpuManager::GetCUDACapableDevicesList()
{
    return cudaCapableDevices_;
}

unsigned int GpuManager::GetCUDACapableDevicesAmount()
{
    int count;
    cudaError_t status = cudaGetDeviceCount(&count);
    if(status != cudaError_t::cudaSuccess)
    {
        auto errorName = cudaGetErrorName(status);
        auto errorDescription = cudaGetErrorString(status);
        LOG_ERROR() << "Failed to get CUDA-capable devices amount. Reason: " << errorName << " - " << errorDescription;
        return 0;
    }
    return count;
}

GPU& GpuManager::SelectMatchingGPU(const std::shared_ptr<Config::JsonConfig> &config)
{
    LOG_TRACE() << "Selecting GPU according to selection policy ...";
    if(!config->Contains(Config::ConfigNodes::ServiceConfig::Gpu))
    {
        LOG_ERROR() << "Service configuration doesn't contain " << Config::ConfigNodes::ServiceConfig::Gpu << " node.";
        throw std::runtime_error("Invalid service configuration.");
    }

    auto gpuConfig = (*config)[Config::ConfigNodes::ServiceConfig::Gpu];

    if(!gpuConfig->Contains(Config::ConfigNodes::ServiceConfig::GpuConfig::SelectionPolicy))
    {
        LOG_ERROR() << "Service GPU configuration doesn't contain "
        << Config::ConfigNodes::ServiceConfig::GpuConfig::SelectionPolicy << " node.";
        throw std::runtime_error("Invalid service GPU configuration.");
    }

    auto gpuSelectionPolicy = (*gpuConfig)[Config::ConfigNodes::ServiceConfig::GpuConfig::SelectionPolicy]->ToString();
    boost::algorithm::trim(gpuSelectionPolicy);
    boost::algorithm::to_upper(gpuSelectionPolicy);


    if (gpuSelectionPolicy == Config::ConfigNodes::ServiceConfig::GpuConfig::Newest)
    {
        LOG_TRACE() << "Selecting GPU with highest compute capability ...";
        auto& selected = *std::max_element(cudaCapableDevices_.begin(), cudaCapableDevices_.end(),
                                 [](GPU& first, GPU& second) {
            return first.computeCapabilityMajor_ * 10 + first.computeCapabilityMinor_ <
            second.computeCapabilityMajor_ * 10 + second.computeCapabilityMinor_;
        });
        LOG_TRACE() << selected.name_ << " with compute capability " << selected.computeCapabilityMajor_
        << "." << selected.computeCapabilityMinor_ << " was selected.";
        return selected;
    }

    if (gpuSelectionPolicy == Config::ConfigNodes::ServiceConfig::GpuConfig::Fastest)
    {
        LOG_TRACE() << "Selecting GPU with highest memory bandwidth ...";
        auto& selected = *std::max_element(cudaCapableDevices_.begin(), cudaCapableDevices_.end(),
                                 [](GPU& first, GPU& second) {
                                     return first.memoryBandwidth_ < second.memoryBandwidth_;
                                 });
        LOG_TRACE() << selected.name_ << " with " << selected.memoryBandwidth_ << " GB/s memory bandwidth was selected.";
        return selected;
    }

    if (gpuSelectionPolicy == boost::algorithm::to_upper_copy(Config::ConfigNodes::ServiceConfig::GpuConfig::MaxSMs))
    {
        LOG_TRACE() << "Selecting GPU with maximal amount of streaming multiprocessors ...";
        auto& selected = *std::max_element(cudaCapableDevices_.begin(), cudaCapableDevices_.end(),
                                 [](GPU& first, GPU& second) {
                                     return first.memoryBandwidth_ < second.memoryBandwidth_;
                                 });
        LOG_TRACE() << selected.name_ << " with " << selected.multiprocessorsAmount_ << " streaming multiprocessors " \
        "was selected.";
        return selected;
    }

    if (gpuSelectionPolicy == Config::ConfigNodes::ServiceConfig::GpuConfig::MostCapacious)
    {
        LOG_TRACE() << "Selecting GPU with maximal size of memory ...";
        auto& selected = *std::max_element(cudaCapableDevices_.begin(), cudaCapableDevices_.end(),
                                 [](GPU& first, GPU& second) {
                                     return first.memoryTotal_ < second.memoryTotal_;
                                 });
        LOG_TRACE() << selected.name_ << " with " << selected.memoryTotal_ << " GB of memory was selected.";
        return selected;
    }

    if (gpuSelectionPolicy == Config::ConfigNodes::ServiceConfig::GpuConfig::Manual)
    {
        LOG_TRACE() << "Selecting GPU by manual-set device id ...";
        if(!gpuConfig->Contains(Config::ConfigNodes::ServiceConfig::GpuConfig::Id))
        {
            LOG_ERROR() << "Service GPU configuration doesn't contain "
            << Config::ConfigNodes::ServiceConfig::GpuConfig::Id << " node.";
            throw std::runtime_error("Invalid service GPU configuration.");
        }
        else
        {
            auto id = (*gpuConfig)[Config::ConfigNodes::ServiceConfig::GpuConfig::Id]->ToInt();
            if (cudaCapableDevices_.size() <= id)
            {
                LOG_ERROR() << "Invalid service GPU configuration. There is no GPU with id=" << id << ".";
                throw std::runtime_error("Invalid service GPU configuration.");
            }
            auto& selected = cudaCapableDevices_[id];
            LOG_TRACE() << selected.name_ << " with device id=" << selected.deviceId_ << " was selected.";
            return selected;
        }
    }

    LOG_ERROR() << "Failed to select matching GPU. Unknown selection policy is configured.";
    throw std::runtime_error("Failed to select matching GPU.");
}

const std::shared_ptr<GPU> &GpuManager::GetCurrentGPU()
{
    return selectedGPU_;
}

}