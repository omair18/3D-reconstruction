#include <chrono>

#include "DatasetCollectingAlgorithm.h"
#include "JsonConfig.h"
#include "Logger.h"
#include "ConfigNodes.h"
#include "ProcessingData.h"

namespace Algorithms
{

DatasetCollectingAlgorithm::DatasetCollectingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
expireTimeoutSeconds_(120)
{
    InitializeInternal(config);
}

bool DatasetCollectingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& datasetUUID = dataset->GetUUID();

    auto& receivedImageDescriptors = const_cast<std::vector<DataStructures::ImageDescriptor>&>(dataset->GetImagesDescriptors());
    unsigned long currentTimestamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    auto existingDataset = datasets_.find(datasetUUID);
    if(existingDataset == datasets_.end())
    {
        LOG_TRACE() << "Started collecting of dataset with UUID " << datasetUUID << ".";
        auto datasetPair = std::make_pair(datasetUUID, std::make_pair(currentTimestamp, processingData));
        processingData->GetModelDataset()->SetProcessingStatus(DataStructures::ModelDataset::ProcessingStatus::COLLECTING);
        datasets_.insert(std::move(datasetPair));
        receivedImageDescriptors.reserve(dataset->GetTotalFramesAmount());
        UpdateExpired(currentTimestamp);
        return false;
    }
    else
    {
        auto& collectedDatasetAndTimestamp = existingDataset->second;
        auto& collectedDataset = collectedDatasetAndTimestamp.second->GetModelDataset();
        auto& collectedDatasetLastUpdateTime = collectedDatasetAndTimestamp.first;
        auto& collectedDatasetImageDescriptors = const_cast<std::vector<DataStructures::ImageDescriptor>&>(collectedDataset->GetImagesDescriptors());

        for (auto& receivedImageDescriptor : receivedImageDescriptors)
        {
            LOG_TRACE() << "Image descriptor " << receivedImageDescriptor.GetFrameId() << "/" << collectedDataset->GetTotalFramesAmount()
            << " was added to the dataset with UUID " << datasetUUID;
            collectedDatasetImageDescriptors.push_back(std::move(receivedImageDescriptor));
        }
        collectedDatasetLastUpdateTime = currentTimestamp;
        UpdateExpired(currentTimestamp);
        if (collectedDataset->GetCurrentFramesAmount() == collectedDataset->GetTotalFramesAmount())
        {
            LOG_TRACE() << "Dataset with UUID " << datasetUUID << " was successfully collected.";
            *processingData = std::move(*collectedDatasetAndTimestamp.second);
            processingData->GetModelDataset()->SetProcessingStatus(DataStructures::ModelDataset::ProcessingStatus::COLLECTED);
            datasets_.erase(datasetUUID);
            return true;
        }
    }
    return false;
}

void DatasetCollectingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void DatasetCollectingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::DatasetCollectingAlgorithm::ExpireTimeout))
    {
        LOG_ERROR() << "Invalid dataset collection algorithm configuration. There must be " <<
                    Config::ConfigNodes::AlgorithmsConfig::DatasetCollectingAlgorithm::ExpireTimeout << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void DatasetCollectingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    LOG_TRACE() << "Initializing " << Config::ConfigNodes::AlgorithmsConfig::AlgorithmsNames::DatasetCollectingAlgorithm
                << " ...";
    ValidateConfig(config);
    expireTimeoutSeconds_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::DatasetCollectingAlgorithm::ExpireTimeout]->ToInt64();
    LOG_TRACE() << "Dataset collection algorithm's expire time out was set to " << expireTimeoutSeconds_ <<".";
}

void DatasetCollectingAlgorithm::UpdateExpired(unsigned long currentTimestamp)
{
    std::vector<std::string> expiredDatasetUUIDs;
    for(auto& [key, value] : datasets_)
    {
        if (value.first + expireTimeoutSeconds_ <= currentTimestamp)
        {
            LOG_TRACE() << "Dataset with UUID " << key << " is expired. Removing it.";
            expiredDatasetUUIDs.push_back(key);
        }
    }
    for(auto& expiredDatasetUUID : expiredDatasetUUIDs)
    {
        datasets_.erase(expiredDatasetUUID);
    }
}

}