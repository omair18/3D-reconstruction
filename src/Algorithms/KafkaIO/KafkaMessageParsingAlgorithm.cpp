#include "KafkaMessageParsingAlgorithm.h"
#include "KafkaMessage.h"
#include "ProcessingData.h"
#include "ImageDescriptor.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "Logger.h"

namespace Algorithms
{

KafkaMessageParsingAlgorithm::KafkaMessageParsingAlgorithm([[maybe_unused]] const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
isInitialized_(false)
{
    InitializeInternal();
}

void KafkaMessageParsingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal();
}

bool KafkaMessageParsingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& message = processingData->GetKafkaMessage();
    if(!message)
    {
        LOG_ERROR() << "Failed to parse kafka message. Message is empty.";
        return false;
    }

    auto& key = message->GetKey();
    if(!key)
    {
        LOG_ERROR() << "Failed to parse kafka message. Message doesn't have a key.";
        return false;
    }

    if(key->Empty())
    {
        LOG_ERROR() << "Failed to parse kafka message. Message's key is empty.";
        return false;
    }

    if (IsImageMessage(key))
    {
        auto cameraId = (*key)[Config::ConfigNodes::MessageNodes::CameraID]->ToInt32();
        if (cameraId < 0)
        {
            LOG_ERROR() << "Failed to parse kafka message. Invalid camera id " << cameraId << ".";
            return false;
        }

        unsigned long timeStamp = (*key)[Config::ConfigNodes::MessageNodes::Timestamp]->ToInt64();
        if(timeStamp <= 0)
        {
            LOG_ERROR() << "Failed to parse kafka message. Invalid time stamp " << timeStamp << ".";
            return false;
        }

        auto frameId = (*key)[Config::ConfigNodes::MessageNodes::FrameID]->ToInt32();
        auto framesTotal = (*key)[Config::ConfigNodes::MessageNodes::FramesTotal]->ToInt32();

        if(framesTotal <= 2)
        {
            LOG_ERROR() << "Failed to parse kafka message. Invalid total frames amount " << framesTotal << ".";
            return false;
        }

        if (frameId < 0 || frameId > framesTotal)
        {
            LOG_ERROR() << "Failed to parse kafka message. Invalid frame ID " << frameId  << ".";
            return false;
        }

        float focalLength;
        float sensorSize;
        int distortionFunctionId;

        if(key->Contains(Config::ConfigNodes::MessageNodes::FocalLength))
        {
            focalLength = (*key)[Config::ConfigNodes::MessageNodes::FocalLength]->ToFloat();
            if(focalLength <= 0)
            {
                LOG_ERROR() << "Failed to parse kafka message. Invalid focal length " << focalLength  << ".";
                return false;
            }
        }
        else
        {
            focalLength = 40;
            LOG_WARNING() << "Focal length of image's camera was not set. Setting it to " << focalLength << "mm.";
        }

        if(key->Contains(Config::ConfigNodes::MessageNodes::SensorSize))
        {
            sensorSize = (*key)[Config::ConfigNodes::MessageNodes::SensorSize]->ToFloat();
            if(sensorSize <= 0)
            {
                LOG_ERROR() << "Failed to parse kafka message. Invalid focal length " << focalLength  << ".";
                return false;
            }
        }
        else
        {
            sensorSize = 33.3f;
            LOG_WARNING() << "Sensor size of image's camera was not set. Setting it to " << sensorSize << "mm.";
        }

        if (key->Contains(Config::ConfigNodes::MessageNodes::DistortionFunctionID))
        {
            distortionFunctionId = (*key)[Config::ConfigNodes::MessageNodes::DistortionFunctionID]->ToInt32();
            if (distortionFunctionId < 0 || distortionFunctionId > 5)
            {
                LOG_ERROR() << "Failed to parse kafka message. Invalid camera's distortion function id " << focalLength  << ".";
                return false;
            }
        }
        else
        {
            distortionFunctionId = 0;
            LOG_WARNING() << "Camera's distortion function id was not set. Setting it to " << distortionFunctionId << ".";
        }

        auto UUID = (*key)[Config::ConfigNodes::MessageNodes::UUID]->ToString();

        DataStructures::ImageDescriptor imageDescriptor;
        auto dataset = std::make_shared<DataStructures::ModelDataset>();

        auto& rawImageData = message->GetData();

        // to avoid copy operation

        auto& modifiableRawImageData = const_cast<std::vector<unsigned char>&>(rawImageData);
        imageDescriptor.SetRawImageData(modifiableRawImageData);

        imageDescriptor.SetCameraId(cameraId);
        imageDescriptor.SetTimestamp(timeStamp);
        imageDescriptor.SetFocalLength(focalLength);
        imageDescriptor.SetSensorSize(sensorSize);
        imageDescriptor.SetFrameId(frameId);
        imageDescriptor.SetCameraDistortionFunctionId(distortionFunctionId);

        dataset->SetUUID(std::move(UUID));
        dataset->SetTotalFramesAmount(framesTotal);

        std::vector<DataStructures::ImageDescriptor> descriptors;
        descriptors.push_back(std::move(imageDescriptor));

        dataset->SetImagesDescriptors(std::move(descriptors));

        processingData->SetModelDataset(std::move(dataset));

    }
    return true;
}

void KafkaMessageParsingAlgorithm::InitializeInternal()
{
    if (!isInitialized_)
    {
        LOG_TRACE() << "Initializing kafka message parsing algorithm ...";
        LOG_TRACE() << "Kafka message parsing algorithm was successfully initialized.";
        isInitialized_ = true;
    }
    else
    {
        LOG_WARNING() << "Kafka message parsing algorithm is already initialized.";
    }
}

bool KafkaMessageParsingAlgorithm::IsImageMessage(const std::shared_ptr<Config::JsonConfig>& messageKey)
{
    return messageKey->Contains(Config::ConfigNodes::MessageNodes::FrameID)
        && messageKey->Contains(Config::ConfigNodes::MessageNodes::FramesTotal)
        && messageKey->Contains(Config::ConfigNodes::MessageNodes::UUID)
        && messageKey->Contains(Config::ConfigNodes::MessageNodes::CameraID)
        && messageKey->Contains(Config::ConfigNodes::MessageNodes::Timestamp);
}

}