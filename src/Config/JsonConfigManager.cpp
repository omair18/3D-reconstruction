#include <memory>
#include <fstream>
#include <string>
#include <iomanip>
#include <filesystem>

#include "JsonConfigManager.h"
#include "JsonConfig.h"
#include "Logger.h"

namespace Config
{

void JsonConfigManager::ReadSettings(const std::filesystem::path & folderPath)
{
    LOG_TRACE() << "Read settings from folder " << folderPath;
    m_folderPath = folderPath.string();
    std::filesystem::directory_iterator fileList(folderPath);

    for (auto const &filename : fileList)
    {
        try
        {
            LOG_TRACE() << "Read file: " << filename.path();
            auto config = std::make_shared<JsonConfig>(filename.path());
            m_configList[filename.path().filename().stem().string()] = config;
        }
        catch (std::exception& ex)
        {
            LOG_ERROR() << ex.what() << " while reading " << filename.path();
        }
    }
}

void JsonConfigManager::ReadSettingsFromFile(const std::filesystem::path& filePath)
{
    LOG_TRACE() << "Read json from file " << filePath;

    try
    {
        auto config = std::make_shared<JsonConfig>(filePath);
        m_configList[filePath.stem().string()] = config;
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << ex.what() << " while reading " << filePath;
    }
}

std::shared_ptr<JsonConfig> JsonConfigManager::GetConfig(const std::string& name)
{
    try
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_configList.at(name);
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << e.what();

        return nullptr;
    }
}

void JsonConfigManager::SetConfig(const std::string & name, const std::shared_ptr<JsonConfig> & config)
{
    try
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_configList[name] = config;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << e.what();
    }
}

std::vector<std::filesystem::path> JsonConfigManager::GetJsonFiles(const std::filesystem::path& folderName)
{
    std::vector<std::filesystem::path> files;

    for (const auto & file : std::filesystem::directory_iterator(folderName))
    {
        if (file.path().extension().string() == jsonExtension)
        {
            files.push_back(file.path());
        }
    }

    return files;
}


void JsonConfigManager::Save(const std::string& key)
{
    try
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::filesystem::path path(m_folderPath);
        path /= (key + jsonExtension);
        std::ofstream outputFile(path, std::ios::out);

        m_configList[key]->Save(outputFile);
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to save config with key " << key << ": " << ex.what();
    }
}

void JsonConfigManager::SaveAll()
{
    SaveAll(m_configList);
}


void JsonConfigManager::SaveAll(const std::map<std::string, std::shared_ptr<JsonConfig>>& configsMap)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto &config : configsMap)
    {
        std::filesystem::path path(m_folderPath);
        path /= (config.first + jsonExtension);
        std::ofstream outputFile(path, std::ios::out);

        outputFile << std::setw(4) << config.second;
    }
}

std::vector<std::string> JsonConfigManager::GetConfigNames()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<std::string> configNames;
    configNames.reserve(m_configList.size());

    for(const auto & [key, value]: m_configList)
    {
        configNames.push_back(key);
    }

    return configNames;
}

}