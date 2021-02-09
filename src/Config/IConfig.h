/**--------------------------------------------------------------------------------------------------
 * @file	Config.h.
 *
 * Declares the IConfig class. A base class for working with configs
 *-----------------------------------------------------------------------------------------------**/

#ifndef ICONFIG_H
#define ICONFIG_H

#include <memory>
#include <vector>

class IConfig
{
public:

    virtual std::shared_ptr<IConfig> operator[] (const std::string & id) = 0;

    virtual std::shared_ptr<IConfig> operator[] (const std::string & id) const = 0;

    virtual std::vector<std::vector<int>> ToVectorVectorInt() = 0;

    virtual std::vector<std::vector<std::vector<int>>> ToVectorVectorVectorInt() = 0;

    virtual std::vector<int> ToVectorInt() = 0;

    virtual std::vector<float> ToVectorFloat() = 0;

    virtual std::vector<std::string> ToVectorString() = 0;

    virtual std::vector<std::vector<double>> ToVectorVectorDouble() = 0;

    virtual bool IsArray() = 0;

    virtual bool IsNull() const = 0;

    virtual std::int32_t ToInt() = 0;

    virtual float ToFloat() = 0;

    virtual double ToDouble() = 0;

    virtual std::wstring ToWString() = 0;

    virtual std::string ToString() = 0;

    virtual bool ToBool() = 0;

    /**
         * @brief Get all objects from current node
         *
         * @return std::vector<std::shared_ptr<IConfig>>
         */
    virtual std::vector<std::shared_ptr<IConfig>> GetObjects() = 0;

    /**
         * @brief Set given node into given branch.
         *
         * @param id 	ID of target branch.
         * @param node 	Node which is nessessary to be setted.
         */
    virtual void SetNode(const std::string& id, std::shared_ptr<IConfig> node) = 0;

    virtual void AddObject(const std::shared_ptr<IConfig> & node) = 0;

    virtual void AddNodeBool(const std::string & key, bool value) = 0;

    virtual void AddNodeFloat(const std::string & key, float val) = 0;

    virtual void AddNodeInt(const std::string&  key, int val) = 0;

    virtual void AddNodeString(const std::string & key, const std::string & value) = 0;

    virtual void AddNodeVecInt(const std::string & key, const std::vector<int> & val) = 0;

    virtual void AddNodeVecFloat(const std::string&  key,const std::vector<float> & val) = 0;

    virtual void AddNodeVecVecInt(const std::string&  key,const std::vector<std::vector<int>> & val) = 0;

    virtual void AddNodeVecVecVecInt(const std::string &  key, const std::vector<std::vector<std::vector<int>>>&  val) = 0;

    virtual void FromJsonString(const std::string & jsonString) = 0;

    /**
      * @brief Sets given configs  to current node.
      *
      * @param configs
      */
    virtual void FromVectorConfigs(const std::vector<std::shared_ptr<IConfig>> & configs) = 0;

    virtual std::string Dump() = 0;

    virtual std::string Dump(int indent) = 0;

    virtual void Save(std::ofstream & stream) = 0;

    virtual ~IConfig() = default;

};

#endif // CONFIG_H

