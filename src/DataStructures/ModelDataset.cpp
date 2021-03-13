#include <unordered_map>

#include "ModelDataset.h"

namespace DataStructures
{

const static std::unordered_map<ModelDataset::ProcessingStatus, const std::string> statusNames =
        {
                { ModelDataset::ProcessingStatus::FAILED, "FAILED" },
                { ModelDataset::ProcessingStatus::COLLECTING, "COLLECTING" },
                { ModelDataset::ProcessingStatus::PROCESSING, "PROCESSING" },
                {ModelDataset::ProcessingStatus::READY, "READY" }
        };

void DataStructures::ModelDataset::SetUUID(const std::string &UUID)
{
    UUID_ = UUID;
}

const std::string &DataStructures::ModelDataset::GetUUID()
{
    return UUID_;
}

std::string DataStructures::ModelDataset::GetProcessingStatusString()
{
    return statusNames.at(status_);
}

DataStructures::ModelDataset::ProcessingStatus DataStructures::ModelDataset::GetProcessingStatus()
{
    return status_;
}

const std::vector<CUDAImageDescriptor> &ModelDataset::GetImagesDescriptors() noexcept
{
    return imagesDescriptors_;
}

}