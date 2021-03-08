/**
 * @file Logger.h.
 *
 * @brief Declares the Logger class. A class for logging the work process of service.
 */


#ifndef LOGGER_H
#define LOGGER_H

#include <boost/log/sources/record_ostream.hpp>

// forward declaration for boost::log::sources::severity_logger_mt
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

/**
 * @namespace Log
 *
 * @brief Namespace of liblog library.
 */
namespace Log
{
    /**
     * @enum SEVERITY_LEVEL
     *
     * @breif Severity level of logging record.
     */
    enum SEVERITY_LEVEL
    {
        /// Debug severity. The lowest severity level.
        DEBUG = 0,

        /// Info severity.
        INFO,

        /// Trace severity.
        TRACE,

        /// Warning severity.
        WARNING,

        /// Error severity.
        ERROR,

        /// Fatal severity. The highest severity level.
        FATAL
    };

/**
 * @class Logger
 *
 * @brief A class for writing logs of service's working process. Singleton class.
 */
class Logger
{
public:

    // forward declaration of RecordStream class.
    class RecordStream;

    /**
     * @brief Provides access to a logger instance.
     *
     * @return A pointer to a logger instance.
     */
    [[nodiscard]] static const Logger* GetInstance() noexcept;

    /**
     * @brief Creates a new record in a log file and provides a stream for adding data to the created record.
     *
     * @return Stream for adding data to the created log record.
     */
    [[nodiscard]] RecordStream CreateRecordSteam() const;

    /**
     * @brief Initializes logger instance.
     *
     * @return True, if logger was initialize successfully. Otherwise returns false.
     */
    static bool Init();

    /**
     * @brief Destroys logger instance.
     */
    static void Free();

private:

    /**
     * @brief Default constructor.
     */
    Logger() = default;

    /**
     * @brief Default destructor.
     */
    ~Logger() = default;

    /**
     * @brief Initializes logging sink.
     *
     * @return True if logging sink was successfully created.
     */
    static bool InitSink();

    /// Initialization status.
    inline static bool initialized_ = false;

    /// A pointer to a logger instance.
    inline static Logger* instance_ = nullptr;

    /// A pointer to a logger backend.
    boost::log::sources::severity_logger_mt<SEVERITY_LEVEL>* backend_ = nullptr;
};

/**
 * @class RecordStream
 *
 * @brief The class of a logging record stream.
 */
class Logger::RecordStream
{
public:

    /**
     * @brief Constructor.
     *
     * @param logger - Pointer to a logger instance for accessing it's backend
     */
    explicit RecordStream(const Logger* logger);

    /**
     * @brief Default destructor.
     */
    ~RecordStream();

    /**
     * @brief Move constructor.
     *
     * @param other - R-value reference to other RecordStream instance.
     */
    RecordStream(RecordStream&& other) noexcept;

    /**
     * @brief Provides stream for writing data to log record.
     *
     * @param severityLevel - Severity level of current record
     * @param functionName - Name of function to use in log record
     * @return Stream for writing data to log record.
     */
    boost::log::record_ostream& GetStream(SEVERITY_LEVEL severityLevel, const std::string& functionName);

    /**
     * @brief Writes collected data from recordPump_ to a log-file.
     */
    void WriteData();

    /**
     * @brief Checks weather record_ is invalid.
     *
     * @return False, if record_ identifies a log record, true, if the record_ is not valid.
     */
    bool operator !();

private:

    /// Pointer to a logger instance_ for accessing it's backend_.
    const Log::Logger* logger_;

    /// Record of the logging file.
    boost::log::record record_;

    /// Record stream data collector.
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
