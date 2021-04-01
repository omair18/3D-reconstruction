#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/expressions/formatters/stream.hpp>
#include <boost/log/expressions/formatters/date_time.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>
#include <boost/log/attributes/clock.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/phoenix/operator.hpp>

#include "Logger.h"

BOOST_LOG_ATTRIBUTE_KEYWORD(function, "Function", std::string)
BOOST_LOG_ATTRIBUTE_KEYWORD(severityLevel, "SeverityLevel", std::string)


constexpr const char* const severityNames[] =
{
        "DEBUG",
        "INFO",
        "TRACE",
        "WARNING",
        "ERROR",
        "FATAL"
};

namespace Log
{
bool Logger::Init()
{
    if(!initialized_)
    {
        instance_ = new Logger();
        instance_->backend_ = new boost::log::sources::severity_logger_mt<SEVERITY_LEVEL>();
        initialized_ = InitSink();
    }
    else
    {
        std::clog << "Failed to initialize logger." << std::endl;
    }
    return initialized_;
}

void Logger::Free()
{
    if(initialized_)
    {
        if(instance_->backend_)
            delete instance_->backend_;
        delete instance_;
        initialized_ = false;
    }
}

const Logger* Logger::GetInstance() noexcept
{
    return instance_;
}

Logger::RecordStream Logger::CreateRecordSteam() const
{
    return Logger::RecordStream(this);
}

bool Logger::InitSink()
{
    typedef boost::log::sinks::synchronous_sink<boost::log::sinks::text_file_backend> fileSink_t;
    auto sink = boost::make_shared<fileSink_t>(
            boost::log::keywords::file_name = "Logs/%Y_%m_%d-%H_%M-%5N.log",
            boost::log::keywords::max_size = 16 * 1024 * 1024,
            boost::log::keywords::auto_flush = true
    );
    if(!sink)
    {
        return false;
    }
    {
        auto backend = sink->locked_backend();

        backend->set_file_collector(boost::log::sinks::file::make_collector(
                boost::log::keywords::target = "Logs",                              // where to store rotated files
                boost::log::keywords::max_size = 16 * 1024 * 1024,                  // maximum total size of the stored files, in bytes
                boost::log::keywords::min_free_space = 100 * 1024 * 1024,           // minimum free space on the drive, in bytes
                boost::log::keywords::max_files = 512                               // maximum number of stored files
        ));

        backend->scan_for_files();
    }

    boost::log::core::get()->add_sink(sink);

    sink->set_formatter(boost::log::expressions::stream
    << boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%d-%m-%Y %H:%M:%S.%f")
    << " [" << boost::this_thread::get_id() << "] "
    << boost::log::expressions::attr<std::string>("SeverityLevel")
    << " %% " << boost::log::expressions::attr<std::string>("Function")
    << ": " << boost::log::expressions::smessage);

    boost::log::attributes::local_clock timeStamp;
    boost::log::core::get()->add_global_attribute("TimeStamp", timeStamp);

    return true;
}

bool Logger::RecordStream::operator!()
{
    return !record_;
}

boost::log::record_ostream& Logger::RecordStream::GetStream(SEVERITY_LEVEL severity, const std::string& functionName)
{
    return (recordPump_->stream()
    << boost::log::add_value("SeverityLevel", severityNames[severity])
    << boost::log::add_value("Function", functionName));
}

Logger::RecordStream::~RecordStream()
{
    if(recordPump_)
    {
        delete recordPump_;
    }
}

Logger::RecordStream::RecordStream(const Logger *logger) :
logger_(logger),
record_(logger->backend_->open_record())
{
    if(!!record_)
    {
        auto recordPump = boost::log::aux::make_record_pump(*logger_->backend_, record_);
        recordPump_ = new auto (boost::move(recordPump));
    }
}

Logger::RecordStream::RecordStream(Logger::RecordStream&& other) noexcept :
logger_(other.logger_),
record_(boost::move(other.record_)),
recordPump_(boost::move(other.recordPump_))
{
}

void Logger::RecordStream::WriteData()
{
    delete recordPump_;
    recordPump_ = nullptr;
    if(!!record_)
    {
        auto recordPump = boost::log::aux::make_record_pump(*logger_->backend_, record_);
        recordPump_ = new auto (boost::move(recordPump));
    }
}

}