#ifndef LOGGER_H
#define LOGGER_H

#include <boost/log/sources/record_ostream.hpp>

namespace boost
{
    BOOST_LOG_OPEN_NAMESPACE
    namespace sources
    {
        template <typename Severity>
        class severity_logger_mt;
    }
    BOOST_LOG_CLOSE_NAMESPACE
}

namespace Log
{
    enum SEVERITY_LEVEL
    {
        DEBUG = 0,
        INFO,
        TRACE,
        WARNING,
        ERROR,
        FATAL
    };

class Logger
{
public:

    class RecordStream;

    [[nodiscard]] static const Logger* GetInstance() noexcept;

    [[nodiscard]] RecordStream CreateRecordSteam() const;

    static bool Init();

    static void Free();

private:
    Logger() = default;

    ~Logger() = default;

    static bool InitSink();

    inline static bool initialized = false;

    inline static Logger* instance = nullptr;
    boost::log::sources::severity_logger_mt<SEVERITY_LEVEL>* backend = nullptr;
};

class Logger::RecordStream
{
public:
    explicit RecordStream(const Logger* logger);
    ~RecordStream();

    RecordStream(RecordStream&& other) noexcept ;
    boost::log::record_ostream& GetStream(SEVERITY_LEVEL severityLevel, const std::string& functionName);

    void WriteData();

    bool operator !();

private:
    const Log::Logger* logger_;
    boost::log::record record_;
    boost::log::aux::record_pump<boost::log::sources::severity_logger_mt<SEVERITY_LEVEL>>* recordPump_ = nullptr;
};

}

#define LOGGER_INIT() Log::Logger::Init()
#define LOGGER_FREE() Log::Logger::Free()

#define LOG_DEBUG() for (Log::Logger::RecordStream stream(Log::Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(Log::SEVERITY_LEVEL::DEBUG, __PRETTY_FUNCTION__)
#define LOG_INFO() for (Log::Logger::RecordStream stream(Log::Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(Log::SEVERITY_LEVEL::INFO, __PRETTY_FUNCTION__)
#define LOG_TRACE() for (Log::Logger::RecordStream stream(Log::Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(Log::SEVERITY_LEVEL::TRACE, __PRETTY_FUNCTION__)
#define LOG_WARNING() for (Log::Logger::RecordStream stream(Log::Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(Log::SEVERITY_LEVEL::WARNING, __PRETTY_FUNCTION__)
#define LOG_ERROR() for (Log::Logger::RecordStream stream(Log::Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(Log::SEVERITY_LEVEL::ERROR, __PRETTY_FUNCTION__)
#define LOG_FATAL() for (Log::Logger::RecordStream stream(Log::Logger::GetInstance()->CreateRecordSteam()); !!stream; stream.WriteData()) stream.GetStream(Log::SEVERITY_LEVEL::FATAL, __PRETTY_FUNCTION__)

#endif //LOGGER_H
