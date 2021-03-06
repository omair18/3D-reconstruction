#include <unordered_map>

#include "ModelDataset.h"
#include "ReconstructionParams.h"

namespace DataStructures
{

const static std::unordered_map<ModelDataset::ProcessingStatus, const std::string> statusNames =
        {
                { ModelDataset::ProcessingStatus::RECEIVED, "RECEIVED"},
                { ModelDataset::ProcessingStatus::FAILED, "FAILED" },
                { ModelDataset::ProcessingStatus::COLLECTING, "COLLECTING" },
                { ModelDataset::ProcessingStatus::COLLECTED, "COLLECTED" },
                { ModelDataset::ProcessingStatus::PROCESSING, "PROCESSING" },
                {ModelDataset::ProcessingStatus::READY, "READY" }
        };

ModelDataset::ModelDataset() :
status_(ProcessingStatus::RECEIVED),
totalSize_(0),
totalFramesAmount_(0),
UUID_(),
imagesDescriptors_(),
reconstructionParams_(std::make_unique<ReconstructionParams>())
{

}

ModelDataset::ModelDataset(const ModelDataset &other)
{
    imagesDescriptors_ = other.imagesDescriptors_;
    UUID_ = other.UUID_;
    totalSize_ = other.totalSize_;
    status_ = other.status_;
    imagesDescriptors_ = other.imagesDescriptors_;
    totalFramesAmount_ = other.totalFramesAmount_;
    if(other.reconstructionParams_)
    {
        reconstructionParams_ = std::make_unique<ReconstructionParams>(*other.reconstructionParams_);
    }
    else
    {
        reconstructionParams_ = std::make_unique<ReconstructionParams>();
    }

};

ModelDataset::ModelDataset(ModelDataset &&other) noexcept :
status_(other.status_),
totalSize_(other.totalSize_),
totalFramesAmount_(other.totalFramesAmount_),
imagesDescriptors_(std::move(other.imagesDescriptors_)),
UUID_(std::move(other.UUID_))
{

}

ModelDataset::~ModelDataset() = default;

ModelDataset& ModelDataset::operator=(const ModelDataset& other)
{
    return *this;
}

void DataStructures::ModelDataset::SetUUID(const std::string &UUID)
{
    UUID_ = UUID;
}

const std::string& DataStructures::ModelDataset::GetUUID() const noexcept
{
    return UUID_;
}

std::string DataStructures::ModelDataset::GetProcessingStatusString() const
{
    return statusNames.at(status_);
}

DataStructures::ModelDataset::ProcessingStatus DataStructures::ModelDataset::GetProcessingStatus() const
{
    return status_;
}

const std::vector<ImageDescriptor>& ModelDataset::GetImagesDescriptors() const noexcept
{
    return imagesDescriptors_;
}

ModelDataset& ModelDataset::operator=(ModelDataset &&other) noexcept
{
    imagesDescriptors_ = std::move(other.imagesDescriptors_);
    reconstructionParams_ = std::move(other.reconstructionParams_);
    UUID_ = std::move(other.UUID_);
    totalFramesAmount_ = other.totalFramesAmount_;
    totalSize_ = other.totalSize_;
    status_ = other.status_;
    return *this;
}

void ModelDataset::SetUUID(std::string &&UUID)
{
    UUID_ = std::move(UUID);
}

void ModelDataset::SetProcessingStatus(ModelDataset::ProcessingStatus status) noexcept
{
    status_ = status;
}

void ModelDataset::SetImagesDescriptors(const std::vector<ImageDescriptor>& imagesDescriptors)
{
    imagesDescriptors_ = imagesDescriptors;
}

void ModelDataset::SetImagesDescriptors(std::vector<ImageDescriptor> &&imagesDescriptors) noexcept
{
    imagesDescriptors_ = std::move(imagesDescriptors);
}

int ModelDataset::GetTotalFramesAmount() const noexcept
{
    return totalFramesAmount_;
}

void ModelDataset::SetTotalFramesAmount(int totalFramesAmount) noexcept
{
    totalFramesAmount_ = totalFramesAmount;
}

int ModelDataset::GetTotalSize() const noexcept
{
    return totalSize_;
}

void ModelDataset::SetTotalSize(int totalSize)
{
    totalSize_ = totalSize;
}

size_t ModelDataset::GetCurrentFramesAmount() const
{
    return imagesDescriptors_.size();
}

const std::unique_ptr<ReconstructionParams>& ModelDataset::GetReconstructionParams() const
{
    return reconstructionParams_;
}

}