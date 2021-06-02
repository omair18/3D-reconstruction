#include "HostToDeviceTransferringAlgorithm.h"
#include "ProcessingData.h"
#include "ImageDescriptor.h"
#include "CUDAImage.h"
#include "Logger.h"

namespace Algorithms
{

HostToDeviceTransferringAlgorithm::HostToDeviceTransferringAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, const std::unique_ptr<GPU::GpuManager>& gpuManager, void* cudaStream) :
IGPUAlgorithm(),
isInitialized_(false),
cudaStream_(cudaStream)
{
    InitializeInternal();
}

bool HostToDeviceTransferringAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
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
                LOG_WARNING() << "Image data is already on the device side. Nothing to transfer.";
            } break;

            case DataStructures::ImageDescriptor::LOCATION::HOST:
            {
                auto& modifiableDescriptor = const_cast<DataStructures::ImageDescriptor&>(descriptor);
                LOG_TRACE() << "Transferring data to host side ...";
                modifiableDescriptor.GetCUDAImage()->MoveFromCvMatAsync(*modifiableDescriptor.GetHostImage(), cudaStream_);
                modifiableDescriptor.SetDataLocation(DataStructures::ImageDescriptor::LOCATION::DEVICE);
            } break;

            case DataStructures::ImageDescriptor::LOCATION::UNDEFINED:
            {
                LOG_ERROR() << "Failed to transfer image data to device side. Undefined data location.";
                return false;
            }

            default:
            {
                LOG_ERROR() << "Failed to transfer image data to device side. Unknown data location.";
                return false;
            }
        }
    }

    return true;
}

void HostToDeviceTransferringAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal();
}

void HostToDeviceTransferringAlgorithm::InitializeInternal()
{
    if (!isInitialized_)
    {
        LOG_TRACE() << "Initializing host to device transferring algorithm ...";
        LOG_TRACE() << "Host to device transferring algorithm was successfully initialized.";
        isInitialized_ = true;
    }
    else
    {
        LOG_WARNING() << "Host to device transferring algorithm is already initialized.";
    }
}

}