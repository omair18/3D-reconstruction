#include <openMVG/sfm/sfm_data.hpp>
#include <openMVG/cameras/cameras.hpp>

#include "DatasetInitializationAlgorithm.h"
#include "ProcessingData.h"
#include "ModelDataset.h"
#include "ReconstructionParams.h"
#include "Logger.h"


namespace Algorithms
{

DatasetInitializationAlgorithm::DatasetInitializationAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
isInitialized_(false)
{
    InitializeInternal();
}

// 0 - Pinhole_Intrinsic
// 1 - Pinhole_Intrinsic_Radial_K1
// 2 - Pinhole_Intrinsic_Radial_K3
// 3 - Pinhole_Intrinsic_Brown_T2
// 4 - Pinhole_Intrinsic_Fisheye

bool DatasetInitializationAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& reconstructionPatams = dataset->GetReconstructionParams();
    auto& datasetUUID = dataset->GetUUID();
    auto& imageDescriptors = dataset->GetImagesDescriptors();

    LOG_TRACE() << "Preparing dataset " << datasetUUID << " for reconstruction ...";

    dataset->SetProcessingStatus(DataStructures::ModelDataset::ProcessingStatus::PROCESSING);

    auto& sfmData = reconstructionPatams->sfMData_;
    auto& intrinsics = sfmData->intrinsics;
    auto& views = sfmData->views;

    for(int i = 0; i < imageDescriptors.size(); ++i)
    {
        auto& imageDescriptor = imageDescriptors[i];
        auto width = imageDescriptor.GetImageWidth();
        auto height = imageDescriptor.GetImageHeight();

        auto principalPointX = width / 2;
        auto principalPointY = height / 2;

        auto focalLengthInMillimeters = imageDescriptor.GetFocalLength();
        auto sensorSizeInMillimeters = imageDescriptor.GetSensorSize();

        auto focalLengthInPixels = static_cast<float>(std::max(width, height)) * (focalLengthInMillimeters / sensorSizeInMillimeters);

        auto distortionFunctionID = imageDescriptor.GetCameraDistortionFunctionId();

        views.insert(std::move(std::make_pair(i, std::make_shared<openMVG::sfm::View>(std::to_string(imageDescriptor.GetFrameId()), i, i, i, width, height))));

        switch (distortionFunctionID)
        {
            case 0:
            {
                intrinsics.insert(std::move(std::make_pair(i, std::make_shared<openMVG::cameras::Pinhole_Intrinsic>(width, height, focalLengthInPixels, principalPointX, principalPointY))));
            } break;

            case 1:
            {
                intrinsics.insert(std::move(std::make_pair(i, std::make_shared<openMVG::cameras::Pinhole_Intrinsic_Radial_K1>(width, height, focalLengthInPixels, principalPointX, principalPointY))));
            } break;

            case 2:
            {
                intrinsics.insert(std::move(std::make_pair(i, std::make_shared<openMVG::cameras::Pinhole_Intrinsic_Radial_K3>(width, height, focalLengthInPixels, principalPointX, principalPointY))));
            } break;

            case 3:
            {
                intrinsics.insert(std::move(std::make_pair(i, std::make_shared<openMVG::cameras::Pinhole_Intrinsic_Brown_T2>(width, height, focalLengthInPixels, principalPointX, principalPointY))));
            } break;

            case 4:
            {
                intrinsics.insert(std::move(std::make_pair(i, std::make_shared<openMVG::cameras::Pinhole_Intrinsic_Fisheye>(width, height, focalLengthInPixels, principalPointX, principalPointY))));
            } break;

            default:
            {
                LOG_ERROR() << "Unknown camera distortion function.";
                dataset->SetProcessingStatus(DataStructures::ModelDataset::ProcessingStatus::FAILED);
                return false;
            }
        }
    }

    LOG_TRACE() << "Dataset " << datasetUUID << " was successfully prepared for reconstruction ...";

    return true;
}

void DatasetInitializationAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal();
}

void DatasetInitializationAlgorithm::InitializeInternal()
{
    if (!isInitialized_)
    {
        LOG_TRACE() << "Dataset initialization algorithm ...";
        LOG_TRACE() << "Dataset initialization algorithm was successfully initialized.";
        isInitialized_ = true;
    }
    else
    {
        LOG_WARNING() << "Dataset initialization algorithm is already initialized.";
    }
}


}