#include <boost/json.hpp>
#include <filesystem>
#include <fstream>

#include "JsonConfig.h"
#include "Logger.h"

JsonConfig::JsonConfig() :
value_(std::make_shared<boost::json::value>())
{

}

JsonConfig::JsonConfig(const JsonConfig& other) :
value_(other.value_)
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
    catch (std::exception ex)
    {
        LOG_ERROR() <<"JSON parsing error: failed to parse " << path;
    }

}

void JsonConfig::AddNodeBool(const std::string &key, bool value)
{
    value_->as_object()[key] = value;
}

void JsonConfig::AddNodeFloat(const std::string &key, float value)
{
    value_->as_object()[key] = value;
}

void JsonConfig::AddNodeInt(const std::string &key, int value)
{
    value_->as_object()[key] = value;
}

void JsonConfig::AddNodeString(const std::string &key, const std::string &value)
{
    value_->as_object()[key] = value;
}

void JsonConfig::AddNodeVecFloat(const std::string &key, const std::vector<float> &value)
{

}

void JsonConfig::AddNodeVecInt(const std::string &key, const std::vector<int> &val)
{

}

void JsonConfig::AddNodeVecVecInt(const std::string &key, const std::vector<std::vector<int>> &val)
{

}

void JsonConfig::AddNodeVecVecVecInt(const std::string &key, const std::vector<std::vector<std::vector<int>>> &val)
{

}

void JsonConfig::AddObject(const std::shared_ptr<IConfig> &node)
{

}

std::string JsonConfig::Dump()
{
    return boost::json::serialize(*value_);
}

std::string JsonConfig::Dump(int indent)
{
    return std::string();
}

void JsonConfig::FromJsonString(const std::string &jsonString)
{
    try
    {
        *value_ = boost::json::parse(jsonString);
    }
    catch (std::exception ex)
    {
        LOG_ERROR() << "JSON parsing error: " << ex.what() << ". Value: " << jsonString;
    }

}

void JsonConfig::FromVectorConfigs(const std::vector<std::shared_ptr<IConfig>> &configs)
{

}

std::vector<std::shared_ptr<IConfig>> JsonConfig::GetObjects()
{
    return std::vector<std::shared_ptr<IConfig>>();
}

bool JsonConfig::IsArray()
{
    return false;
}

bool JsonConfig::IsNull() const
{
    return false;
}

void JsonConfig::Save(std::ofstream &stream)
{

}

void JsonConfig::SetNode(const std::string &id, std::shared_ptr<IConfig> node)
{

}

bool JsonConfig::ToBool()
{
    bool result = false;
    try
    {
        result = value_->as_bool();
        return result;
    }
    catch (std::exception &e)
    {
        LOG_ERROR() << "Failed to convert value " << value_ << " to bool: " << e.what();
        return result;
    }
}

double JsonConfig::ToDouble()
{
    double result = 0;
    try
    {
        result = value_->as_double();
        return result;
    }
    catch (std::exception &e)
    {
        LOG_ERROR() << "Failed to convert value " << value_ << " to double: " << e.what();
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
    catch (std::exception &e)
    {
        LOG_ERROR() << "Failed to convert value " << value_ << " to float: " << e.what();
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
    catch (std::exception &e)
    {
        LOG_ERROR() << "Failed to convert value " << value_ << " to int32_t: " << e.what();
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
    catch (std::exception &e)
    {
        LOG_ERROR() << "Failed to convert value " << value_ << " to int32_t: " << e.what();
        return result;
    }
}

std::vector<float> JsonConfig::ToVectorFloat()
{
    std::vector<float> result;
    try
    {
        return result;
    }
    catch (std::exception &e)
    {
        LOG_ERROR() << "Failed to convert value " << value_ << " to int32_t: " << e.what();
        return result;
    }
}

std::vector<int> JsonConfig::ToVectorInt()
{
    return std::vector<int>();
}

std::vector<std::string> JsonConfig::ToVectorString()
{
    return std::vector<std::string>();
}

std::vector<std::vector<double>> JsonConfig::ToVectorVectorDouble()
{
    return std::vector<std::vector<double>>();
}

std::vector<std::vector<int>> JsonConfig::ToVectorVectorInt()
{
    return std::vector<std::vector<int>>();
}

std::vector<std::vector<std::vector<int>>> JsonConfig::ToVectorVectorVectorInt()
{
    return std::vector<std::vector<std::vector<int>>>();
}

std::wstring JsonConfig::ToWString()
{
    return std::wstring();
}

std::shared_ptr<IConfig> JsonConfig::operator[](const std::string &id)
{
    return std::shared_ptr<IConfig>();
}

std::shared_ptr<IConfig> JsonConfig::operator[](const std::string &id) const
{
    return std::shared_ptr<IConfig>();
}

