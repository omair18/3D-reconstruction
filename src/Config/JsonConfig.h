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


class JsonConfig : public IConfig
{
public:

    JsonConfig();

    JsonConfig(const JsonConfig& other);

    JsonConfig(JsonConfig&& other) noexcept;

    explicit JsonConfig(const std::filesystem::path& path);

    ~JsonConfig() override = default;

    void AddNodeBool(const std::string &key, bool value) override;

    void AddNodeFloat(const std::string &key, float val) override;

    void AddNodeInt(const std::string &key, int val) override;

    void AddNodeString(const std::string &key, const std::string &value) override;

    void AddNodeVecFloat(const std::string &key, const std::vector<float> &val) override;

    void AddNodeVecInt(const std::string &key, const std::vector<int> &val) override;

    void AddNodeVecVecInt(const std::string &key, const std::vector<std::vector<int>> &val) override;

    void AddNodeVecVecVecInt(const std::string &key, const std::vector<std::vector<std::vector<int>>> &val) override;

    void AddObject(const std::shared_ptr<IConfig> &node) override;

    std::string Dump() override;

    std::string Dump(int indent) override;

    void FromJsonString(const std::string &jsonString) override;

    void FromVectorConfigs(const std::vector<std::shared_ptr<IConfig>> &configs) override;

    std::vector<std::shared_ptr<IConfig>> GetObjects() override;

    bool IsArray() override;

    bool IsNull() const override;

    void Save(std::ofstream &stream) override;

    void SetNode(const std::string &id, std::shared_ptr<IConfig> node) override;

    bool ToBool() override;

    double ToDouble() override;

    float ToFloat() override;

    std::int32_t ToInt() override;

    std::string ToString() override;

    std::vector<float> ToVectorFloat() override;

    std::vector<int> ToVectorInt() override;

    std::vector<std::string> ToVectorString() override;

    std::vector<std::vector<double>> ToVectorVectorDouble() override;

    std::vector<std::vector<int>> ToVectorVectorInt() override;

    std::vector<std::vector<std::vector<int>>> ToVectorVectorVectorInt() override;

    std::wstring ToWString() override;

    std::shared_ptr<IConfig> operator[] (const std::string & id) override;

    std::shared_ptr<IConfig> operator[] (const std::string & id) const override;

private:

    std::shared_ptr<boost::json::value> value_;
};

#endif // JSON_CONFIG_H
