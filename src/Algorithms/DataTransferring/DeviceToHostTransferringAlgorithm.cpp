#include "DeviceToHostTransferringAlgorithm.h"
#include "ProcessingData.h"
#include "ImageDescriptor.h"
#include "CUDAImage.h"
#include "Logger.h"

namespace Algorithms
{

DeviceToHostTransferringAlgorithm::DeviceToHostTransferringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
isInitialized_(false),
cudaStream_(cudaStream)
{
    InitializeInternal();
}

void DeviceToHostTransferringAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal();
}

bool DeviceToHostTransferringAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& descriptors = dataset->GetImagesDescriptors();
    auto& datasetUUID = dataset->GetUUID();

    for(auto& descriptor : descriptors)
    {
        auto dataLocation = descriptor.GetDataLocation();

        LOG_TRACE() << "Processing image descriptor " << descriptor.GetFrameId() << "/"
        << dataset->GetTotalFramesAmount() << " of dataset " << datasetUUID;

        switch (dataLocation)
        {
            case DataStructures::ImageDescriptor::LOCATION::DEVICE:
            {
                auto& modifiableDescriptor = const_cast<DataStructures::ImageDescriptor&>(descriptor);
                LOG_TRACE() << "Transferring data to host side ...";
                modifiableDescriptor.GetCUDAImage()->MoveToCvMatAsync(*modifiableDescriptor.GetHostImage(), cudaStream_);
                modifiableDescriptor.SetDataLocation(DataStructures::ImageDescriptor::LOCATION::HOST);
            } break;

            case DataStructures::ImageDescriptor::LOCATION::HOST:
            {
                LOG_WARNING() << "Image data is already on the host side. Nothing to transfer.";
            } break;

            case DataStructures::ImageDescriptor::LOCATION::UNDEFINED:
            {
                LOG_ERROR() << "Failed to transfer image data to host side. Undefined data location.";
                return false;
            }

            default:
            {
                LOG_ERROR() << "Failed to transfer image data to host side. Unknown data location.";
                return false;
            }
        }
    }

    return true;
}

void DeviceToHostTransferringAlgorithm::InitializeInternal()
{
    if (!isInitialized_)
    {
        LOG_TRACE() << "Initializing device to host transferring algorithm ...";
        LOG_TRACE() << "Device to host transferring algorithm was successfully initialized.";
        isInitialized_ = true;
    }
    else
    {
        LOG_WARNING() << "Device to host transferring algorithm is already initialized.";
    }
}


}
