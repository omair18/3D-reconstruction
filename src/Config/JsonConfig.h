/**
 * @file JsonConfig.h.
 *
 * @brief Declares the JsonConfig class. A class for working with configs in JSON format.
 */

#ifndef JSON_CONFIG_H
#define JSON_CONFIG_H

#include <vector>

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

// forward declaration for boost::json::value
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
 * @brief Namespace of libconfig library.
 */
namespace Config
{

/**
 * @class JsonConfig
 *
 * @brief A class for working with configs in JSON format.
 */
class JsonConfig
{
public:

    /**
     * @brief Default constructor.
     */
    JsonConfig();

    /**
     * @brief Copy constructor. Creates a copy of other-param object.
     *
     * @param other - Instance of existing JsonConfig
     */
    JsonConfig(const JsonConfig& other);

    /**
     * @brief Move constructor. Moves data from other-param to current instance_.
     *
     * @param other - Instance of existing JsonConfig
     */
    JsonConfig(JsonConfig&& other) noexcept;

    /**
     * @brief Constructs new JsonConfig instance_ with data from file, located at path-param.
     *
     * @param path - Path to .json file
     */
    explicit JsonConfig(const std::filesystem::path& path);

    /**
     * @brief Constructs new JsonConfig instance_ with data from existing JSON-object.
     *
     * @param object - Instance of existing JSON-object
     */
    explicit JsonConfig(const boost::json::value& object);

    /**
     * @brief Default destructor.
     */
    ~JsonConfig() = default;

    /**
     * @brief Adds bool value-param with key key-param. If current JsonConfig is no an object,
     * does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param value - Value of boolean to add
     */
    void AddNodeBool(const std::string &key, bool value);

    /**
     * @brief Adds float value-param with key key-param. If current JsonConfig is no an object,
     * does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param val - Value of float to add
     */
    void AddNodeFloat(const std::string& key, float val);

    /**
     * @brief Adds int value-param with key key-param. If current JsonConfig is no an object,
     * does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param val - Value of integer to add
     */
    void AddNodeInt(const std::string& key, int val);

    /**
     * @brief Adds string value-param with key key-param. If current JsonConfig is no an object,
     * does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param value - Value of string to add
     */
    void AddNodeString(const std::string& key, const std::string &value);

    /**
     * @brief Adds vector&lt;float&gt; value-param with key key-param. If current JsonConfig is no an object,
     * does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param val - Array of float values to add
     */
    void AddNodeVecFloat(const std::string& key, const std::vector<float>& val);

    /**
     * @brief Adds vector&lt;int&gt; value-param with key key-param. If current JsonConfig is no an object,
     * does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param val - Array of integral values to add
     */
    void AddNodeVecInt(const std::string& key, const std::vector<int>& val);

    /**
     * @brief Adds vector&lt;vector&lt;int&gt;&gt; value-param with key key-param.
     * If current JsonConfig is no an object, does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param val - 2D-array of integral values to add
     */
    void AddNodeVecVecInt(const std::string& key, const std::vector<std::vector<int>>& val);

    /**
     * @brief Adds vector&lt;vector&lt;vector&lt;int&gt&gt;&gt; value-param with key key-param.
     * If current JsonConfig is no an object, does nothing and creates a record in log file with severity ERROR.
     *
     * @param key - Key value
     * @param val - 3D-array of integral values to add
     */
    void AddNodeVecVecVecInt(const std::string& key, const std::vector<std::vector<std::vector<int>>>& val);

    /**
     * @brief Converts object from object-param to JSON value and adds it to array.
     * If current JsonConfig is no an array, does nothing and creates a record in log file with severity ERROR.
     *
     * @param object - An object for adding to array
     */
    void AddObject(const std::shared_ptr<JsonConfig>& object);

    /**
     * @brief Converts current JSON value at JsonConfig to string.
     *
     * @return Current JSON value as string.
     */
    std::string Dump();

    /**
     * @brief Initializes current JsonConfig instance_ with JSON from jsonString-param.
     * If the instance_ of JsonConfig was not empty, old data will be replaced with new data.
     * If jsonString-param contains invalid JSON, does nothing and puts a record in a log file with
     * severity ERROR about parsing failure.
     *
     * @param jsonString - String with JSON value
     */
    void FromJsonString(const std::string& jsonString);

    /**
     * @brief Initializes current JsonConfig instance_ with array of JSON values from configs-param.
     * If the instance_ of JsonConfig was not empty, old data will be replaced with new data.
     * If one of the object from configs-param contains invalid JSON, skips invalid object and puts a record in a
     * log file with severity ERROR about parsing failure.
     *
     * @param configs - Vector of JSON values
     */
    void FromVectorConfigs(const std::vector<std::shared_ptr<JsonConfig>>& configs);

    /**
     * @brief Converts current JSON object/array to vector of all nodes it has.
     *
     * @return Vector of all the nodes in current JSON value.
     */
    std::vector<std::shared_ptr<JsonConfig>> GetObjects();

    /**
     * @brief Checks weather current JSON value is an array.
     *
     * @return True if current JSON value is an array. Otherwise returns false.
     */
    bool IsArray();

    /**
     * @brief Checks weather current JSON value is null.
     *
     * @return True if current JSON value is null. Otherwise returns false.
     */
    bool IsNull() const;

    /**
     * @brief Checks weather current JSON object contains node with key-param key.
     *
     * @param key - Key to check
     * @return True if contains. Otherwise returns false
     */
    bool Contains(const std::string& key);

    /**
     * @brief Puts current JSON value to ostream as string in a human-readable format.
     *
     * @param stream - Output data stream
     */
    void Save(std::ofstream& stream);

    /**
     * @brief Converts value at node-param to JSON value and sets it to current object at node id-param.
     * If node-param contains invalid JSON object, does nothing and puts a record to log file with ERROR severity.
     *
     * @param key - Key for adding a new node
     * @param object - JSON value for adding to current JSON object with key key-param
     */
    void SetNode(const std::string& key, const std::shared_ptr<JsonConfig>& object);

    /**
     * @brief Converts value at current node to bool. If the value at current node is not a bool,
     * returns false and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as bool or false.
     */
    bool ToBool();

    /**
     * @brief Converts value at current node to double. If the value at current node is not a double,
     * returns 0 and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as double or 0.
     */
    double ToDouble();

    /**
     * @brief Converts value at current node to float. If the value at current node is not a float,
     * returns 0 and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as float or 0.
     */
    float ToFloat();

    /**
     * @brief Converts value at current node to int32_t. If the value at current node is not an int32_t,
     * returns 0 and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as int32_t or 0
     */
    std::int32_t ToInt();

    /**
     * @brief Converts value at current node to string. If the value at current node is not a string,
     * returns empty string and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as string or an empty string.
     */
    std::string ToString();

    /**
     * @brief Converts value at current node to vector&lt;float&gt;. If the value at current node is not a
     * vector&lt;float&gt;, returns empty vector and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as vector&lt;float&gt; or an empty vector.
     */
    std::vector<float> ToVectorFloat();

    /**
     * @brief Converts value at current node to vector&lt;int&gt;. If the value at current node is not a
     * vector&lt;int&gt;, returns empty vector and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as vector&lt;int&gt; or an empty vector.
     */
    std::vector<int> ToVectorInt();

    /**
     * @brief Converts value at current node to vector&lt;string&gt;. If the value at current node is not a
     * vector&lt;string&gt;, returns empty vector and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as vector&lt;string&gt; or an empty vector.
     */
    std::vector<std::string> ToVectorString();

    /**
     * @brief Converts value at current node to vector&lt;vector&lt;double&gt;&gt;. If the value at current node is
     * not a vector&lt;vector&lt;double&gt;&gt;, returns empty vector and puts a record to log file with
     * ERROR severity.
     *
     * @return Value of current node as vector&lt;vector&lt;double&gt;&gt; or an empty vector.
     */
    std::vector<std::vector<double>> ToVectorVectorDouble();

    /**
     * @brief Converts value at current node to vector&lt;vector&lt;int&gt;&gt;. If the value at current node is
     * not a vector&lt;vector&lt;int&gt;&gt;, returns empty vector and puts a record to log file with
     * ERROR severity.
     *
     * @return Value of current node as vector&lt;vector&lt;int&gt;&gt; or an empty vector.
     */
    std::vector<std::vector<int>> ToVectorVectorInt();

    /**
     * @brief Converts value at current node to vector&lt;vector&lt;vector&lt;int&gt;&gt;&gt;.
     * If the value at current node is not a vector&lt;vector&lt;vector&lt;int&gt;&gt;&gt;, returns empty vector and
     * puts a record to log file with ERROR severity.
     *
     * @return Value of current node as vector&lt;vector&lt;vector&lt;int&gt;&gt;&gt; or an empty vector.
     */
    std::vector<std::vector<std::vector<int>>> ToVectorVectorVectorInt();

    /**
     * @brief Converts value at current node to wstring. If the value at current node is not a (w)string,
     * returns empty wstring and puts a record to log file with ERROR severity.
     *
     * @return Value of current node as wstring or an empty wstring.
     */
    std::wstring ToWString();

    /**
     * @brief Provides access to node at current JSON value with key key-param as a pointer.
     * If there is no such node, returns nullptr and puts a record to a log file with severity ERROR.
     *
     * @param key - Key of the node to access
     * @return A pointer to key-param node or nullptr.
     */
    std::shared_ptr<JsonConfig> operator[] (const std::string& key);

    /**
     * @brief Provides access to node at current JSON value with key key-param as a pointer.
     * If there is no such node, returns nullptr and puts a record to a log file with severity ERROR.
     *
     * @param key - Key of the node to access
     * @return A pointer to key-param node or nullptr.
     */
    std::shared_ptr<JsonConfig> operator[] (const std::string& key) const;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    JsonConfig& operator=(const JsonConfig& other);

    /**
     * @brief
     *
     * @param other
     * @return
     */
    JsonConfig& operator=(JsonConfig&& other) noexcept;

    /**
     * @brief
     *
     * @param other
     * @return
     */
    bool operator==(const JsonConfig& other);

private:

    /// A pointer to JSON data.
    std::shared_ptr<boost::json::value> value_;
};

}
#endif // JSON_CONFIG_H
