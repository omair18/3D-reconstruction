#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include "CUDAImageDecodingAlgorithm.h"
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

CUDAImageDecodingAlgorithm::CUDAImageDecodingAlgorithm(const std::shared_ptr<Config::JsonConfig> &config, const std::unique_ptr<GPU::GpuManager> &gpuManager, void *cudaStream) :
IGPUAlgorithm(),
removeSourceData_(false),
decoders_(),
cudaStream_(cudaStream)
{
    InitializeInternal(config);
}

bool CUDAImageDecodingAlgorithm::Process(const std::shared_ptr<DataStructures::ProcessingData> &processingData)
{
    auto& dataset = processingData->GetModelDataset();
    auto& imageDescriptors = dataset->GetImagesDescriptors();
    auto& datasetUUID = dataset->GetUUID();
    auto totalFramesInDataset = dataset->GetTotalFramesAmount();

    if(imageDescriptors.empty())
    {
        LOG_ERROR() << "CUDA image decoding error. There is nothing to decode.";
        return false;
    }

    bool decodingStatus = true;
    for(auto& imageDescriptor : imageDescriptors)
    {
        auto& image = imageDescriptor.GetCUDAImage();
        if(decodingStatus)
        {
            LOG_TRACE() << "Decoding image " << imageDescriptor.GetFrameId() << "/" << totalFramesInDataset
            << " of dataset " << datasetUUID << " ...";
            auto& rawImageData = imageDescriptor.GetRawImageData();
            decodingStatus = false;

            if (rawImageData.empty())
            {
                LOG_ERROR() << "CUDA image decoding error. There is nothing to decode.";
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
                << " of dataset " << dataset->GetUUID() << " was successful.";
                if(removeSourceData_)
                {
                    auto& modifiableImageDescriptor = const_cast<DataStructures::CUDAImageDescriptor&>(imageDescriptor);
                    modifiableImageDescriptor.SetRawImageData({});
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

void CUDAImageDecodingAlgorithm::Initialize(const std::shared_ptr<Config::JsonConfig> &config)
{
    InitializeInternal(config);
}

void CUDAImageDecodingAlgorithm::ValidateConfig(const std::shared_ptr<Config::JsonConfig> &config)
{
    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::CUDAImageDecodingAlgorithmConfig::Decoders))
    {
        LOG_ERROR() << "Invalid CUDA image decoder algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::CUDAImageDecodingAlgorithmConfig::Decoders << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }

    if(!config->Contains(Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::RemoveSourceData))
    {
        LOG_ERROR() << "Invalid CUDA image decoder algorithm configuration. There must be "
        << Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::RemoveSourceData
        << " node in algorithm configuration.";
        throw std::runtime_error("Invalid algorithm configuration.");
    }
}

void CUDAImageDecodingAlgorithm::InitializeInternal(const std::shared_ptr<Config::JsonConfig> &config)
{
    if(decoders_.empty())
    {
        LOG_TRACE() << "Initializing CUDA image decoding algorithm ...";
        ValidateConfig(config);

        removeSourceData_ = (*config)[Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::RemoveSourceData]->ToBool();
        auto decodersNamesList = (*config)[Config::ConfigNodes::AlgorithmsConfig::CpuImageDecodingAlgorithmConfig::Decoders]->ToVectorString();

        for(auto& decoderName : decodersNamesList)
        {
            boost::algorithm::trim(decoderName);
            boost::algorithm::to_upper(decoderName);
            auto decoderType = ConvertDecoderNameToDecodingType(decoderName);
            auto decoder = Decoding::ImageDecoderFactory::Create(decoderType, true, cudaStream_);
            if(decoder)
            {
                decoder->Initialize();
                if(decoder->IsInitialized())
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
        LOG_TRACE() << "CUDA image decoding algorithm was successfully initialized.";
    }
    else
    {
        LOG_WARNING() << "CUDA image decoding algorithm is already initialized.";
    }
}

}