#ifndef JSON_CONFIG_H
#define JSON_CONFIG_H

#include <bits/fs_fwd.h>

#include "IConfig.h"

namespace boost
{
    namespace json
    {
        class value;
    }
}

/**
 * @namespace Config
 *
 * @brief Namespace of libconfig library
 */
namespace Config
{

/**
 *
 */
class JsonConfig : public IConfig
{
public:

    /**
     * tested
     */
    JsonConfig();

    /**
     *
     * @param other
     */
    JsonConfig(const JsonConfig& other);

    /**
     *
     * @param other
     */
    JsonConfig(JsonConfig&& other) noexcept;

    /**
     * tested
     * @param path
     */
    explicit JsonConfig(const std::filesystem::path& path);

    /**
     *
     * @param node
     */
    explicit JsonConfig(const boost::json::value& node);

    /**
     * @brief Default destructor.
     */
    ~JsonConfig() override = default;

    /**
     *
     * @param key
     * @param value
     */
    void AddNodeBool(const std::string &key, bool value) override;

    /**
     *
     * @param key
     * @param val
     */
    void AddNodeFloat(const std::string &key, float val) override;

    /**
     *
     * @param key
     * @param val
     */
    void AddNodeInt(const std::string &key, int val) override;

    /**
     *
     * @param key
     * @param value
     */
    void AddNodeString(const std::string &key, const std::string &value) override;

    /**
     *
     * @param key
     * @param val
     */
    void AddNodeVecFloat(const std::string &key, const std::vector<float> &val) override;

    /**
     *
     * @param key
     * @param val
     */
    void AddNodeVecInt(const std::string &key, const std::vector<int> &val) override;

    /**
     *
     * @param key
     * @param val
     */
    void AddNodeVecVecInt(const std::string &key, const std::vector<std::vector<int>> &val) override;

    /**
     *
     * @param key
     * @param val
     */
    void AddNodeVecVecVecInt(const std::string &key, const std::vector<std::vector<std::vector<int>>> &val) override;

    /**
     *
     * @param node
     */
    void AddObject(const std::shared_ptr<IConfig> &node) override;

    /**
     *
     * @return
     */
    std::string Dump() override;

    /**
     *
     * @param jsonString
     */
    void FromJsonString(const std::string &jsonString) override;

    /**
     *
     * @param configs
     */
    void FromVectorConfigs(const std::vector<std::shared_ptr<IConfig>> &configs) override;

    /**
     *
     * @return
     */
    std::vector<std::shared_ptr<IConfig>> GetObjects() override;

    /**
     *
     * @return
     */
    bool IsArray() override;

    /**
     *
     * @return
     */
    bool IsNull() const override;

    /**
     *
     * @param stream
     */
    void Save(std::ofstream &stream) override;

    /**
     *
     * @param id
     * @param node
     */
    void SetNode(const std::string &id, std::shared_ptr<IConfig> node) override;

    /**
     *
     * @return
     */
    bool ToBool() override;

    /**
     *
     * @return
     */
    double ToDouble() override;

    /**
     *
     * @return
     */
    float ToFloat() override;

    /**
     *
     * @return
     */
    std::int32_t ToInt() override;

    /**
     *
     * @return
     */
    std::string ToString() override;

    /**
     *
     * @return
     */
    std::vector<float> ToVectorFloat() override;

    /**
     *
     * @return
     */
    std::vector<int> ToVectorInt() override;

    /**
     *
     * @return
     */
    std::vector<std::string> ToVectorString() override;

    /**
     *
     * @return
     */
    std::vector<std::vector<double>> ToVectorVectorDouble() override;

    /**
     *
     * @return
     */
    std::vector<std::vector<int>> ToVectorVectorInt() override;

    /**
     *
     * @return
     */
    std::vector<std::vector<std::vector<int>>> ToVectorVectorVectorInt() override;

    /**
     *
     * @return
     */
    std::wstring ToWString() override;

    /**
     *
     * @param id
     * @return
     */
    std::shared_ptr<IConfig> operator[] (const std::string & id) override;

    /**
     *
     * @param id
     * @return
     */
    std::shared_ptr<IConfig> operator[] (const std::string & id) const override;

private:

    ///
    std::shared_ptr<boost::json::value> value_;
};

}
#endif // JSON_CONFIG_H
