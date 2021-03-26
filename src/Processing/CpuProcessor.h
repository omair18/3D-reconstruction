#ifndef CPU_PROCESSOR_H
#define CPU_PROCESSOR_H

#include "IProcessor.h"

namespace Networking
{
    class KafkaConsumer;
    class KafkaProducer;
}

namespace Processing
{

class CpuProcessor : public IProcessor
{
public:
    explicit CpuProcessor(const std::shared_ptr<Config::JsonConfig>& config);

    ~CpuProcessor() override;

    void Process() override;

    void Initialize() override;

private:

};

}
#endif // CPU_PROCESSOR_H
