#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include "CpuImageDecodingAlgorithm.h"
#include "ImageDecoderFactory.h"
#include "JsonConfig.h"
#include "ConfigNodes.h"
#include "Logger.h"
#include "ProcessingData.h"
#include "IImageDecoder.h"

static Decoding::DecoderType ConvertDecoderNameToDecodingType(const std::string& decoderName)
{
    if(decoderName == Config::ConfigNodes::DecodingConfig::DecodersNames::OpenCV)
    {
        return Decoding::DecoderType::OPENCV_IMAGE_DECODER;
    }

    if(decoderName == Config::ConfigNodes::DecodingConfig::DecodersNames::NvJPEG)
    {
        return Decoding::DecoderType::NVJPEG_IMAGE_DECODER;
    }

    if(decoderName == Config::ConfigNodes::DecodingConfig::DecodersNames::NvJPEGHardware)
    {
        return Decoding::DecoderType::NVJPEG_HARDWARE_IMAGE_DECODER;
    }

    if(decoderName == Config::ConfigNodes::DecodingConfig::DecodersNames::NvJPEG2K)
    {
        return Decoding::DecoderType::NVJPEG2K_IMAGE_DECODER;
    }

    return Decoding::DecoderType::UNKNOWN_DECODER;
}

namespace Algorithms
{

Algorithms::CpuImageDecodingAlgorithm::CpuImageDecodingAlgorithm(const std::shared_ptr<Config::JsonConfig>& config, [[maybe_unused]] const std::unique_ptr<GPU::GpuManager>& gpuManager, [[maybe_unused]] void* cudaStream) :
ICPUAlgorithm(),
removeSourceData_(false),
decoders_()
{
    InitializeInternal(config);
}

bool CpuImageDecodingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData>& processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& imageDescriptors = dataset->GetImagesDescriptors();
    auto& datasetUUID = dataset->GetUUID();
    auto totalFramesInDataset = dataset->GetTotalFramesAmount();

    if(imageDescriptors.empty())
    {
        LOG_ERROR() << "CPU image decoding error. There is nothing to decode.";
        return false;
    }

    bool decodingStatus = true;
    for(auto& imageDescriptor : imageDescriptors)
    {
        if(decodingStatus)
        {
            LOG_TRACE() << "Decoding image " << imageDescriptor.GetFrameId() << "/" << totalFramesInDataset
            << " of dataset " << datasetUUID << " ...";
            auto& rawImageData = imageDescriptor.GetRawImageData();
            auto& image = imageDescriptor.GetHostImage();
            decodingStatus = false;

            if (rawImageData.empty())
            {
                LOG_ERROR() << "CPU image decoding error. There is nothing to decode.";
                return false;
            }

            for(auto& decoder : decoders_)
            {
                decodingStatus = decoder->Decode(rawImageData.data(), rawImageData.size(), *image);
                if(decodingStatus)
                {
                    break;
                }
            }

            if(decodingStatus)
            {
                LOG_TRACE() << "Decoding image " << imageDescriptor.GetFrameId() << "/" << totalFramesInDataset
                << " of dataset " << datasetUUID << " was successful.";
                auto& modifiableImageDescriptor = const_cast<DataStructures::ImageDescriptor&>(imageDescriptor);
                modifiableImageDescriptor.SetDataLocation(DataStructures::ImageDescriptor::LOCATION::HOST);
                if(removeSourceData_)
                {
                    modifiableImageDescriptor.ClearRawImageData();
                }
            }
            else
            {
                LOG_TRACE() << "Failed to decode image " << imageDescriptor.GetFrameId() << "/"
                << totalFramesInDataset << " of dataset " << datasetUUID << ".";
            }
        }
        else
        {
            LOG_ERROR() << "Previous image decoding was failed. Skipping remaining data.";
            return false;
        }
    }

    return decodingStatus;

}

void CpuImageDecodingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig>& config)
{
    InitializeInternal(config);
}

void CpuImageDecodingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::Decoders))
    {
        LOG_ERROR() << "Invalid CPU image decoder algorithm configuration. There must be " <<
        Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::Decoders << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::RemoveSourceData))
    {
        LOG_ERROR() << "Invalid CPU image decoder algorithm configuration. There must be " <<
                    Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::RemoveSourceData << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void CpuImageDecodingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig>& config)
{
    if(decoders_.empty())
    {
        LOG_TRACE() << "Initializing CPU image decoding algorithm ...";
        ValidateConfig(config);

        removeSourceData_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::RemoveSourceData]->ToBool();
        auto decodersNamesList = (*config)[Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::Decoders]->ToVectorString();

        for(auto& decoderName : decodersNamesList)
        {
            boost::algorithm::trim(decoderName);
            boost::algorithm::to_upper(decoderName);
            auto decoderType = ConvertDecoderNameToDecodingType(decoderName);
            auto decoder = Decoding::ImageDecoderFactory::Create(decoderType, false);
            if(decoder)
            {
                decoder->Initialize();
                if (decoder->IsInitialized())
                {
                    decoders_.push_back(std::move(decoder));
                }
            }
            else
            {
                LOG_ERROR() << "Failed to create " << decoderName;
                throw std::runtime_error("Invalid algorithm configuration.");
            }
        }
        LOG_TRACE() << "CPU image decoding algorithm was successfully initialized.";
    }
    else
    {
        LOG_WARNING() << "CPU image decoding algorithm is already initialized.";
    }
}

}