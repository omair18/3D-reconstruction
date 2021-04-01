#include <boost/json.hpp>
#include <filesystem>
#include <fstream>

#include "JsonConfig.h"
#include "Logger.h"

namespace Config
{

JsonConfig::JsonConfig() :
value_(std::make_shared<boost::json::value>())
{
    value_->emplace_object();
}

JsonConfig::JsonConfig(const JsonConfig& other) :
value_(std::make_shared<boost::json::value>(*other.value_))
{

}

JsonConfig::JsonConfig(JsonConfig&& other) noexcept :
value_(std::move(other.value_))
{

}

JsonConfig::JsonConfig(const std::filesystem::path& path)
{
    try
    {
        boost::json::error_code errorCode;
        std::ifstream configFile(path.string());
        boost::json::stream_parser parser;
        std::string line;
        while(std::getline(configFile,line))
        {
            parser.write(line,errorCode );
            if(errorCode)
            {
                LOG_ERROR() << "JSON parse error: failed to parse line: " << line << "; Error code: " << errorCode;
                return;
            }
        }
        parser.finish(errorCode);
        if(errorCode)
        {
            LOG_ERROR() << "JSON parse error: failed create json value. Error code: " << errorCode;
            return;
        }

        value_ = std::make_shared<boost::json::value>(parser.release());
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() <<"JSON parsing error: failed to parse " << path;
    }

}

JsonConfig::JsonConfig(const boost::json::value& object) :
value_(std::make_shared<boost::json::value>(object))
{

}

void JsonConfig::AddNodeBool(const std::string& key, bool value)
{
    try
    {
        value_->as_object()[key] = value;
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to add value " << value << " with key " << key << " to " << (*value_);
    }
}

void JsonConfig::AddNodeFloat(const std::string& key, float value)
{
    try
    {
        value_->as_object()[key] = value;
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to add value " << value << " with key " << key << " to " << (*value_);
    }
}

void JsonConfig::AddNodeInt(const std::string& key, int value)
{
    try
    {
        value_->as_object()[key] = value;
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to add value " << value << " with key " << key << " to " << (*value_);
    }
}

void JsonConfig::AddNodeString(const std::string& key, const std::string& value)
{
    try
    {
        value_->as_object()[key] = value;
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to add value " << value << " with key " << key << " to " << (*value_);
    }
}

void JsonConfig::AddNodeVecFloat(const std::string& key, const std::vector<float>& value)
{
    boost::json::array array(value.begin(), value.end());
    value_->as_object()[key] = array;
}

void JsonConfig::AddNodeVecInt(const std::string& key, const std::vector<int>& value)
{
    boost::json::array array(value.begin(), value.end());
    value_->as_object()[key] = array;
}

void JsonConfig::AddNodeVecVecInt(const std::string& key, const std::vector<std::vector<int>>& value)
{
    boost::json::array array;
    array.reserve(value.size());
    for (auto& vector : value)
    {
        array.push_back(boost::json::array(vector.begin(), vector.end()));
    }
    value_->as_object()[key] = array;
}

void JsonConfig::AddNodeVecVecVecInt(const std::string& key, const std::vector<std::vector<std::vector<int>>>& value)
{
    boost::json::array array;
    array.reserve(value.size());
    for (auto& vector : value)
    {
        boost::json::array internalArray;
        internalArray.reserve(vector.size());
        for(auto& vec : vector)
        {
            internalArray.push_back(boost::json::array(vec.begin(), vec.end()));
        }
        array.push_back(internalArray);
    }
    value_->as_object()[key] = array;
}

void JsonConfig::AddObject(const std::shared_ptr<JsonConfig>& object)
{
    if (value_->is_array())
    {
        auto json = boost::json::parse(object->Dump());
        value_->as_array().push_back(json);
    }
    else
    {
        LOG_ERROR() << "Node is not array" << object->Dump();
    }
}

std::string JsonConfig::Dump()
{
    try
    {
        return boost::json::serialize(*value_);
    }
    catch (std::exception& exception)
    {
        LOG_ERROR() << "Failed to serialize JSON value to string. Details: " << exception.what();
        return "";
    }

}

void JsonConfig::FromJsonString(const std::string& jsonString)
{
    try
    {
        *value_ = boost::json::parse(jsonString);
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "JSON parsing error: " << ex.what() << ". Value: " << jsonString;
    }

}

void JsonConfig::FromVectorConfigs(const std::vector<std::shared_ptr<JsonConfig>>& configs)
{
    value_->emplace_array();

    for (const auto & config : configs)
    {
        try
        {
            auto json = boost::json::parse(config->Dump());
            value_->as_array().push_back(json);
        }
        catch (std::exception& ex)
        {
            LOG_ERROR() << "JSON parsing error: " << config->Dump();
        }
    }

}

std::vector<std::shared_ptr<JsonConfig>> JsonConfig::GetObjects()
{
    std::vector<std::shared_ptr<JsonConfig>> result;

    if(value_->is_object())
    {
        auto& value = value_->as_object();
        for (auto& object : value)
        {
            result.push_back(std::make_shared<JsonConfig>(object.value()));
        }
    }
    else if (value_->is_array())
    {
        auto& value = value_->as_array();
        for (auto& object : value)
        {
            result.push_back(std::make_shared<JsonConfig>(object));
        }
    }

    return result;

}

bool JsonConfig::IsArray()
{
    return value_->is_array();
}

bool JsonConfig::IsNull() const
{
    return value_->is_null();
}

bool JsonConfig::Contains(const std::string& key)
{
    return value_->as_object().contains(key);
}

void JsonConfig::Save(std::ofstream& stream)
{
    const std::string spacer = "    ";
    int nestedLevel = 0;
    auto jsonString = Dump();
    for (auto character : jsonString)
    {
        switch (character)
        {
            case '{':
            case '[':
            {
                ++nestedLevel;
                stream << character << std::endl;
                for(int i = 0; i < nestedLevel; ++i)
                {
                    stream << spacer;
                }
            } break;
            case '}':
            case ']':
            {
                stream << std::endl;
                --nestedLevel;
                for(int i = 0; i < nestedLevel; ++i)
                {
                    stream << spacer;
                }
                stream << character;
            } break;
            case ':':
            {
                stream <<": ";
            } break;
            case ',':
            {
                stream << ", " << std::endl;
                for(int i = 0; i < nestedLevel; ++i)
                {
                    stream << spacer;
                }
            } break;
            default:
            {
                stream << character;
            } break;
        }

    }
}

void JsonConfig::SetNode(const std::string& key, const std::shared_ptr<JsonConfig>& object)
{
    try
    {
        auto json = boost::json::parse(object->Dump());
        value_->as_object()[key] = json;
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to set node " << object->Dump() << " to " << key;
    }
}

bool JsonConfig::ToBool()
{
    bool result = false;
    try
    {
        result = value_->as_bool();
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to bool: " << e.what();
        return result;
    }
}

double JsonConfig::ToDouble()
{
    double result = 0;
    try
    {
        result = value_->to_number<double>();
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to double: " << e.what();
        return result;
    }
}

float JsonConfig::ToFloat()
{
    float result = 0;
    try
    {
        result = value_->to_number<float>();
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to float: " << e.what();
        return result;
    }
}

std::int32_t JsonConfig::ToInt()
{
    std::int32_t result = 0;
    try
    {
        result = value_->to_number<std::int32_t>();
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to int32_t: " << e.what();
        return result;
    }
}

std::string JsonConfig::ToString()
{
    std::string result;
    try
    {
        result = value_->as_string().data();
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to string: " << e.what();
        return result;
    }
}

std::vector<float> JsonConfig::ToVectorFloat()
{
    std::vector<float> result;
    try
    {
        auto& array = value_->as_array();
        result.reserve(array.size());
        for(auto& number : array)
        {
            result.push_back(number.to_number<float>());
        }
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to vector<float>: " << e.what();
        return result;
    }
}

std::vector<int> JsonConfig::ToVectorInt()
{
    std::vector<int> result;
    try
    {
        auto& array = value_->as_array();
        result.reserve(array.size());
        for(auto& number : array)
        {
            result.push_back(number.to_number<int>());
        }
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to vector<int>: " << e.what();
        return result;
    }
}

std::vector<std::string> JsonConfig::ToVectorString()
{
    std::vector<std::string> result;
    try
    {
        auto& array = value_->as_array();
        result.reserve(array.size());
        for(auto& string : array)
        {
            result.emplace_back(string.as_string().data());
        }
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to vector<string>: " << e.what();
        return result;
    }
}

std::vector<std::vector<double>> JsonConfig::ToVectorVectorDouble()
{
    std::vector<std::vector<double>> result;
    try
    {
        auto& array = value_->as_array();
        result.reserve(array.size());
        for(auto& arrayValue : array)
        {
            std::vector<double> internalArray;
            auto& internalArrayValue = arrayValue.as_array();
            internalArray.reserve(internalArrayValue.size());
            for(auto& number : internalArrayValue)
            {
                internalArray.push_back(number.as_double());
            }
            result.push_back(std::move(internalArray));
        }
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to vector<vector<double>>: " << ex.what();
    }
    return result;

}

std::vector<std::vector<int>> JsonConfig::ToVectorVectorInt()
{
    std::vector<std::vector<int>> result;
    try
    {
        auto& array = value_->as_array();
        result.reserve(array.size());
        for(auto& arrayValue : array)
        {
            std::vector<int> internalArray;
            auto& internalArrayValue = arrayValue.as_array();
            internalArray.reserve(internalArrayValue.size());
            for(auto& number : internalArrayValue)
            {
                internalArray.push_back(number.as_int64());
            }
            result.push_back(std::move(internalArray));
        }
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to vector<vector<int>>: " << ex.what();
    }
    return result;
}

std::vector<std::vector<std::vector<int>>> JsonConfig::ToVectorVectorVectorInt()
{
    std::vector<std::vector<std::vector<int>>> result;
    try
    {
        auto& array = value_->as_array();
        result.reserve(array.size());
        for(auto& arrayValue : array)
        {
            std::vector<std::vector<int>> internalArray;
            auto& internalArrayValue = arrayValue.as_array();
            internalArray.reserve(internalArrayValue.size());
            for(auto& internalArrayValueLayer2 : internalArrayValue)
            {
                auto& internalArrayValueLayer2Array = internalArrayValueLayer2.as_array();
                std::vector<int> numbers;
                numbers.reserve(internalArrayValueLayer2Array.size());
                for(auto& number : internalArrayValueLayer2Array)
                {
                    numbers.push_back(number.as_int64());
                }
                internalArray.push_back(std::move(numbers));
            }
            result.push_back(std::move(internalArray));
        }
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to vector<vector<vector<int>>>: " << ex.what();
    }
    return result;
}

std::wstring JsonConfig::ToWString()
{
    std::wstring result;
    try
    {
        std::string data = value_->as_string().data();
        result = std::wstring(data.begin(), data.end());
        return result;
    }
    catch (std::exception& e)
    {
        LOG_ERROR() << "Failed to convert value " << (*value_) << " to int32_t: " << e.what();
        return result;
    }
}

std::shared_ptr<JsonConfig> JsonConfig::operator[](const std::string &key)
{
    try
    {
        return std::make_shared<JsonConfig>(value_->at(key));
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "operator [] has got a wrong argument: " << key;
        return nullptr;
    }
}

std::shared_ptr<JsonConfig> JsonConfig::operator[](const std::string& key) const
{
    try
    {
        return std::make_shared<JsonConfig>(value_->at(key));
    }
    catch (std::exception& ex)
    {
        LOG_ERROR() << "operator [] has got a wrong argument: " << key;
        return nullptr;
    }
}

JsonConfig &JsonConfig::operator=(const JsonConfig &other)
{
    if(this == &other)
    {
        return *this;
    }
    value_ = std::make_shared<boost::json::value>(*other.value_);
    return *this;
}

JsonConfig &JsonConfig::operator=(JsonConfig &&other) noexcept
{
    if(other.value_ == value_)
    {
        return *this;
    }
    value_ = std::move(other.value_);
    return *this;
}

bool JsonConfig::operator==(const JsonConfig &other)
{
    if (this == &other)
    {
        return true;
    }

    if (value_ == other.value_)
    {
        return true;
    }

    return *value_ == *other.value_;
}

}
