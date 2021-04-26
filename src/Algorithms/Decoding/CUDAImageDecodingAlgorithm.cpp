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
    auto& imageDescriptors = dataset.GetImagesDescriptors();

    if(imageDescriptors.empty())
    {
        LOG_ERROR() << "CUDA image decoding error. There is nothing to decode.";
        return false;
    }

    bool decodingStatus = true;
    for(auto& imageDescriptor : imageDescriptors)
    {
        if(decodingStatus)
        {
            LOG_TRACE() << "Decoding image " << imageDescriptor.GetFrameId() << "/" << dataset.GetTotalFramesAmount()
            << " of dataset " << dataset.GetUUID() << " ...";
            auto& rawImageData = imageDescriptor.GetRawImageData();
            DataStructures::CUDAImage image;
            decodingStatus = false;

            if (rawImageData.empty())
            {
                LOG_ERROR() << "CUDA image decoding error. There is nothing to decode.";
                return false;
            }

            for(auto& decoder : decoders_)
            {
                decodingStatus = decoder->Decode(rawImageData.data(), rawImageData.size(), image);
                if(decodingStatus)
                {
                    break;
                }
            }

            if(decodingStatus)
            {
                LOG_TRACE() << "Decoding image " << imageDescriptor.GetFrameId() << "/" << dataset.GetTotalFramesAmount()
                << " of dataset " << dataset.GetUUID() << " was successful.";
                imageDescriptor.Set
                if(removeSourceData_)
                {
                    imageDescriptorModifiable.SetRawImageData({});
                }
            }
            else
            {
                LOG_TRACE() << "Failed to decode image " << imageDescriptor.GetFrameId() << "/"
                << dataset.GetTotalFramesAmount() << " of dataset " << dataset.GetUUID() << ".";
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
                decoders_.push_back(std::move(decoder));
            }
            else
            {
                LOG_ERROR() << "Failed to create " << decoderName;
                throw std::runtime_error("Invalid algorithm configuration.");
            }
        }
    }
    else
    {
        LOG_WARNING() << "CUDA image decoding algorithm is already initialized.";
    }
}

}