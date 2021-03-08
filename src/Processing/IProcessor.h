#ifndef INTERFACE_PROCESSOR_H
#define INTERFACE_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>

namespace DataStructures
{
    class ProcessingData;
    template <typename T>
    class ProcessingQueue;
}

namespace Algorithms
{
    class IAlgorithm;
}

namespace Config
{
    class JsonConfig;
}

namespace Processing
{

class IProcessor
{
public:
    explicit IProcessor(const std::shared_ptr<Config::JsonConfig>& config);

    virtual void Process() = 0;

    [[nodiscard]] const std::string& GetName() const
    {
        return name_;
    }

    void SetName(const std::string& name)
    {
        name_ = name;
    };

    virtual ~IProcessor() = default;

protected:

    std::string name_;

    std::shared_ptr<DataStructures::ProcessingQueue<std::shared_ptr<DataStructures::ProcessingData>>> inputQueue_;

    std::shared_ptr<DataStructures::ProcessingQueue<std::shared_ptr<DataStructures::ProcessingData>>> outputQueue_;

    std::vector<std::unique_ptr<Algorithms::IAlgorithm>> processingAlgorithms_;
};


}

#endif // INTERFACE_PROCESSOR_H
