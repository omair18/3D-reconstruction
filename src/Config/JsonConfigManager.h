/**
 * @file JsonConfigManager.h.
 *
 * @brief Declares the JsonConfigManager class. This class works with settings for service and its components.
 */

#ifndef JSON_CONFIG_MANAGER_H
#define JSON_CONFIG_MANAGER_H

#include <mutex>
#include <vector>
#include <map>

// forward declaration for std::filesystem::path
namespace std
{
    _GLIBCXX_BEGIN_NAMESPACE_VERSION

    namespace filesystem
    {
#if _GLIBCXX_USE_CXX11_ABI
        inline namespace __cxx11 __attribute__((__abi_tag__ ("cxx11"))) { }
#endif
        _GLIBCXX_BEGIN_NAMESPACE_CXX11
        class path;
        _GLIBCXX_END_NAMESPACE_CXX11
    }
    _GLIBCXX_END_NAMESPACE_VERSION
}

/**
 * @namespace Config
 *
 * @brief Namespace of libconfig library.
 */
namespace Config
{

// forward declaration for Config::JsonConfig
class JsonConfig;

/**
 * @class JsonConfigManager
 *
 * @brief This class works with settings for service and its components.
 */
class JsonConfigManager
{

public:

    /**
     * @brief Default constructor.
     */
    JsonConfigManager() = default;

    /**
    * @brief Finds all json files in a folder, reads them and creates a map of configs where key is the name of
    * json file.
    *
    * @param folderPath - Path to the folder where json configs must be
    */
    void ReadSettings(const std::filesystem::path& folderPath);

    /**
    * @brief Reads json file and adds configs from it to map where key is the name of this file.
    *
    * @param filePath - Path to json file with configs
    */
    void ReadSettingsFromFile(const std::filesystem::path& filePath);

    /**
     * @brief
     *
     * @param configName
     * @return
     */
    bool ConfigExists(const std::string& configName);

    /**
     * @brief Provides access to config with key name-param in configList_. If there is no such key in map,
     * returns nullptr and puts a record in a log file with ERROR severity.
     *
     * @param name - Key of config in configs map
     * @return Pointer to config in configList_ with key name-param. Returns nullptr in failure case.
     */
    std::shared_ptr<JsonConfig> GetConfig(const std::string& name);

    /**
    * @brief Retrieves all config names from the map.
    *
    * @return Vector of map's keys.
    */
    std::vector<std::string> GetConfigNames();

    /**
    * @brief Sets given config to given key in map.
    *
    * @param name - Key of config to set
    * @param config - Pointer to JsonConfig object
    */
    void SetConfig(const std::string& name, const std::shared_ptr<JsonConfig>& config);

    /**
    * @brief Saves configs by given name to file folderPath_/key.json. In failure case does nothing and puts
    * a record in log file with ERROR severity.
    *
    * @param key - Key of config in configList_
    */
    void Save(const std::string& key);

    /**
    * @brief Saves all configs from configList_ to files at folderPath_.
    *
    */
    void SaveAll();

    /**
    * @brief Saves all configs from given map to files at folderPath_.
    *
    * @param configsMap - Map of configs to save
    */
    void SaveAll(const std::map<std::string, std::shared_ptr<JsonConfig>>& configsMap);

    /**
     * @brief Default destructor.
     */
    ~JsonConfigManager() = default;

private:

    /**
     * @brief Find all .json files at folderName-param and returns a vector of paths for this files.
     *
     * @param folderName - Path to search JSON files
     * @return Vector of paths to .json files.
     */
    std::vector<std::filesystem::path> GetJsonFiles(const std::filesystem::path& folderName);

    /// Extension of files to work with.
    const std::string jsonExtension = ".json";

    /// Container for configs' names and pointers to them.
    std::map<std::string, std::shared_ptr<JsonConfig>> configList_;

    /// Mutex to provide thread-safety.
    std::mutex mutex_;

    /// Path of folder to save configs.
    std::string folderPath_;
};

}

#endif // JSON_CONFIG_MANAGER_H