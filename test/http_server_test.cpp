#ifndef BOOST_BEAST_EXAMPLE_COMMON_SERVER_CERTIFICATE_HPP
#define BOOST_BEAST_EXAMPLE_COMMON_SERVER_CERTIFICATE_HPP

#include <boost/asio/buffer.hpp>
#include <boost/asio/ssl/context.hpp>
#include <cstddef>
#include <memory>

/*  Load a signed certificate into the ssl context, and configure
    the context for use with a server.

    For this to work with the browser or operating system, it is
    necessary to import the "Beast Test CA" certificate into
    the local certificate store, browser, or operating system
    depending on your environment Please see the documentation
    accompanying the Beast certificate for more details.
*/
inline
void
load_server_certificate(boost::asio::ssl::context& ctx)
{
    /*
        The certificate was generated from CMD.EXE on Windows 10 using:

        winpty openssl dhparam -out dh.pem 2048
        winpty openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 10000 -out cert.pem -subj "//C=US\ST=CA\L=Los Angeles\O=Beast\CN=www.example.com"
    */

    std::string const cert =
            "-----BEGIN CERTIFICATE-----\n"
            "MIIEBTCCAu2gAwIBAgIUIp/OArzA/iYi3rGcmWQr8f5XvIIwDQYJKoZIhvcNAQEL\n"
            "BQAwgZExCzAJBgNVBAYTAkJZMQ4wDAYDVQQIDAVNaW5zazEOMAwGA1UEBwwFTWlu\n"
            "c2sxDjAMBgNVBAoMBUJTVUlSMQ0wCwYDVQQLDARGQ1NOMR4wHAYDVQQDDBVQaG90\n"
            "b2dyYW1tZXRyeSBzZXJ2ZXIxIzAhBgkqhkiG9w0BCQEWFHZhbGVyYWtsOTlAZ21h\n"
            "aWwuY29tMB4XDTIxMDIwMzIyNDEzMVoXDTI2MDIwMjIyNDEzMVowgZExCzAJBgNV\n"
            "BAYTAkJZMQ4wDAYDVQQIDAVNaW5zazEOMAwGA1UEBwwFTWluc2sxDjAMBgNVBAoM\n"
            "BUJTVUlSMQ0wCwYDVQQLDARGQ1NOMR4wHAYDVQQDDBVQaG90b2dyYW1tZXRyeSBz\n"
            "ZXJ2ZXIxIzAhBgkqhkiG9w0BCQEWFHZhbGVyYWtsOTlAZ21haWwuY29tMIIBIjAN\n"
            "BgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAlPHIV5mqLXVk9NkLrlE7UbaYduMR\n"
            "BqHWdT/qV6LX2+qJiFYh1SNwQqCrNwbURqZzFaUJISZcV+s824CiDpTtSM1FUYSb\n"
            "Otb1jIV2M75xjLw88acZn6a7kuXzTTERb11fRZc2goC9qCDKzK7VByzzePojXFS2\n"
            "+goyF6tKpZxVf6peEofIl/YGVkSjDmEMiHl0Kuzltz6vTP4LsCV/6uYjr2hPiNvy\n"
            "iYGjjYJuKxYo+D3aiAg82F4hDRfq/tPOX7TqHbaL95ruQ41qfAhyI4d39mUWGxgD\n"
            "pZ/ZG8MW4oRXtjQnObKFKy00TwDqfCFnQNuJx/lg0FfJ1BOfF8cx9DtZlwIDAQAB\n"
            "o1MwUTAdBgNVHQ4EFgQUgK18Fl3Gdn6VlxmEy12jIW7ifrwwHwYDVR0jBBgwFoAU\n"
            "gK18Fl3Gdn6VlxmEy12jIW7ifrwwDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0B\n"
            "AQsFAAOCAQEAcLa+gy4/rNV10OP26I7nti16i+xVoTwh9cp7ae10htQj2HXOn5aL\n"
            "J38JHT9f82KynzqAGyDp9LH0AWe/wuX0fYLKMW1URp12sfy5ViD40U9YL9/vF5Qn\n"
            "P2gfTRYxauRK4BZlEXFifSwrVshwnJ8PAC7shaCOfO4qrB0YZq3xwFtOl6FW7Ozp\n"
            "8urAy42S3XZLGvIU02N8ouimMxs7z98GzkQKnQUlqSUZIVlED/Qvgqbhet4BQwWE\n"
            "ccLbuWx9wAm+szDap/HQZu8FhA6A5ct+RZifBCog6V1lK/82hNZ1rW8R8lsgmMBb\n"
            "CZZk9lLIZumNs1rgJHV7cf5upunJWMYQDg==\n"
            "-----END CERTIFICATE-----\n";

    std::string const key =
    "-----BEGIN PRIVATE KEY-----\n"
    "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCU8chXmaotdWT0\n"
    "2QuuUTtRtph24xEGodZ1P+pXotfb6omIViHVI3BCoKs3BtRGpnMVpQkhJlxX6zzb\n"
    "gKIOlO1IzUVRhJs61vWMhXYzvnGMvDzxpxmfpruS5fNNMRFvXV9FlzaCgL2oIMrM\n"
    "rtUHLPN4+iNcVLb6CjIXq0qlnFV/ql4Sh8iX9gZWRKMOYQyIeXQq7OW3Pq9M/guw\n"
    "JX/q5iOvaE+I2/KJgaONgm4rFij4PdqICDzYXiENF+r+085ftOodtov3mu5DjWp8\n"
    "CHIjh3f2ZRYbGAOln9kbwxbihFe2NCc5soUrLTRPAOp8IWdA24nH+WDQV8nUE58X\n"
    "xzH0O1mXAgMBAAECggEAOPCNpF+MXUCJzcjD3GoVgVGKy7hWo/Buqfg7QDFy0BHD\n"
    "bilvWanomrANhEXKFRBA4r9q1A90MfAuHcP5MBXi+Hfwzg8hwMa4rHXBEFr30dOM\n"
    "gN1ewFffUXMzJgliHu7VIyeBoTZSStnubX1Q1yNqGI+XwQ5pvQD00XOcX996GAmW\n"
    "DCQG+1+2OHuAsAtUjZEFyszgRyE8r0QuCw48aYrjIQh9liF1i41N9Q6KHI2KWQGm\n"
    "56N9wnFja/MnsxIjq10yTcXG4CgQl2z9DbcYBNy1QRTyso/PojIGDoBRCRnAFXed\n"
    "MXbFbXhcfkrICDknwyhBbr4KQ0bkn/cpN1QAkspOEQKBgQDDwLlO4GA6Dff4uUaf\n"
    "CXrpkvZjJiTZbv9spiUerM0yh6/v/4h6tUOU2g0GjeKJ0WJLIH9T3GfV9P8cglRW\n"
    "DDQAo2VJMoKbVQ/JKkGC2OjUsCjZb+UUlKzgYXvwYD8+UdMxS2duQ6BvFe2fC7Qj\n"
    "EsRl5Pe27q4IF+p70GIG6lQO/QKBgQDCyQyMUeOkbjX4N9bCLFCt/YW2qGHfq9YQ\n"
    "KMuj7VJW5/bW2uUzEpAnPaQMQS66ctFX2LmYLnfBgZnxwZ0YLjg+2VKqDaxvdxvW\n"
    "9NVTBNgfZ2aP2QLOL1H+6hLW4FQBWNsj0ZoLZYoDDiBpQ+fHid4ePHrmNxK8ngUe\n"
    "JP07e0CRIwKBgQCaa4cDTQc75NcP04B9ZuoaJf/NuUJqcCB4ohw4NZLt4JIO3mLY\n"
    "gfOdEeIioHzrwUwr/afUDNOwucMhx0ImWxwOwNbexkS3qoN4aqghQ2FwnsKqvJVX\n"
    "Be3WV7CAsDxTuhLDGTuvhROjbgX0aDQjUuOxic1L9r88zTkBkSIzWFkVNQKBgFvs\n"
    "JU/TFWrz1+Rb9a2M3EY/7mpU7Ftx7IOTvQaPaNw6e/LJOoDvsbfTDyf5WuQCa2jZ\n"
    "jjyTMVDQqKW+iTRwG3sNdEVsToQL4cd/o2yaTKQFWBFqiDhlfHygWmOtWfeIx2IP\n"
    "+HKIaKkocYM72JYCKiB6ykT6mI0Kxb+EFT2M3NuhAoGAOWcP5G09sGVqqDCV0uFE\n"
    "fpUfa6WBSFO3HlCnJgzxTXpZuN2zbqt+JB0mYDfTb5aaqr4iKr9dvbWQ2fZnmJ5k\n"
    "P4HHLXexwrLmSu7sOjsTwGVH+oB2izGTBGdQcZoEPK2rOWa4hpVw2c7JwtMTC+vP\n"
    "FhTd4jGrfId3LmV965QE4QI=\n"
    "-----END PRIVATE KEY-----\n";

    std::string const dh =
    "-----BEGIN DH PARAMETERS-----\n"
    "MIIBCAKCAQEA3PukmFtFov4Ub8Mz2vC0BLD3/mTCstoibJRnBizETVIdlmJk0c15\n"
    "K/0d+Em76rD+I2Z/5+r1/KQ5DINHX8vlIXO85VR2doyRAarQAjpvVOmf1vYTMlka\n"
    "O8kgqWlcrtavlVwydb3d/F2rWtMi6/oWT3mH7LVA7zy4X7EQ2rphrq7KD1n2mTHK\n"
    "UTMh4LNQliETrmDAaWIS8AtnApMI09K0j98PgTpImKG0g6y3OqGnXxbwWfngmCJ+\n"
    "aqwPq5RXAY0txNCpOkTIM3nBU5NmKgxXFxR1pf+LT394bfvGZZpIkGGApEiPguG1\n"
    "Xvht3knGMqvU6fUA1zyNLGVd3jHpfZq6IwIBAg==\n"
    "-----END DH PARAMETERS-----\n";

    ctx.set_password_callback(
            [](std::size_t,
               boost::asio::ssl::context_base::password_purpose)
            {
                return "test";
            });

    ctx.set_options(
            boost::asio::ssl::context::default_workarounds |
            boost::asio::ssl::context::no_sslv2 |
            boost::asio::ssl::context::single_dh_use);

    ctx.use_certificate_chain(
            boost::asio::buffer(cert.data(), cert.size()));

    ctx.use_private_key(
            boost::asio::buffer(key.data(), key.size()),
            boost::asio::ssl::context::file_format::pem);

    ctx.use_tmp_dh(
            boost::asio::buffer(dh.data(), dh.size()));
}

#endif

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/signal_set.hpp>
#include <boost/asio/strand.hpp>
#include <boost/make_unique.hpp>
#include <boost/optional.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace beast = boost::beast;                 // from <boost/beast.hpp>
namespace http = beast::http;                   // from <boost/beast/http.hpp>
namespace net = boost::asio;                    // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;               // from <boost/asio/ssl.hpp>
using tcp = boost::asio::ip::tcp;               // from <boost/asio/ip/tcp.hpp>

// Return a reasonable mime type based on the extension of a file.
beast::string_view
mime_type(beast::string_view path)
{
    using beast::iequals;
    auto const ext = [&path]
    {
        auto const pos = path.rfind(".");
        if(pos == beast::string_view::npos)
            return beast::string_view{};
        return path.substr(pos);
    }();
    if(iequals(ext, ".htm"))  return "text/html";
    if(iequals(ext, ".html")) return "text/html";
    if(iequals(ext, ".php"))  return "text/html";
    if(iequals(ext, ".css"))  return "text/css";
    if(iequals(ext, ".txt"))  return "text/plain";
    if(iequals(ext, ".js"))   return "application/javascript";
    if(iequals(ext, ".json")) return "application/json";
    if(iequals(ext, ".xml"))  return "application/xml";
    if(iequals(ext, ".swf"))  return "application/x-shockwave-flash";
    if(iequals(ext, ".flv"))  return "video/x-flv";
    if(iequals(ext, ".png"))  return "image/png";
    if(iequals(ext, ".jpe"))  return "image/jpeg";
    if(iequals(ext, ".jpeg")) return "image/jpeg";
    if(iequals(ext, ".jpg"))  return "image/jpeg";
    if(iequals(ext, ".gif"))  return "image/gif";
    if(iequals(ext, ".bmp"))  return "image/bmp";
    if(iequals(ext, ".ico"))  return "image/vnd.microsoft.icon";
    if(iequals(ext, ".tiff")) return "image/tiff";
    if(iequals(ext, ".tif"))  return "image/tiff";
    if(iequals(ext, ".svg"))  return "image/svg+xml";
    if(iequals(ext, ".svgz")) return "image/svg+xml";
    return "application/text";
}

// Append an HTTP rel-path to a local filesystem path.
// The returned path is normalized for the platform.
std::string
path_cat(
        beast::string_view base,
        beast::string_view path)
{
    if(base.empty())
        return std::string(path);
    std::string result(base);
    char constexpr path_separator = '/';
    if(result.back() == path_separator)
        result.resize(result.size() - 1);
    result.append(path.data(), path.size());
    return result;
}

// This function produces an HTTP response for the given
// request. The type of the response object depends on the
// contents of the request, so the interface requires the
// caller to pass a generic lambda for receiving the response.
template<
        class Body, class Allocator,
        class Send>
void
handle_request(
        beast::string_view doc_root,
        http::request<Body, http::basic_fields<Allocator>>&& req,
        Send&& send)
{
    // Returns a bad request response
    auto const bad_request =
            [&req](beast::string_view why)
            {
                http::response<http::string_body> res{http::status::bad_request, req.version()};
                res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
                res.set(http::field::content_type, "text/html");
                res.keep_alive(req.keep_alive());
                res.body() = std::string(why);
                res.prepare_payload();
                return res;
            };

    // Returns a not found response
    auto const not_found =
            [&req](beast::string_view target)
            {
                http::response<http::string_body> res{http::status::not_found, req.version()};
                res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
                res.set(http::field::content_type, "text/html");
                res.keep_alive(req.keep_alive());
                res.body() = "The resource '" + std::string(target) + "' was not found.";
                res.prepare_payload();
                return res;
            };

    // Returns a server error response
    auto const server_error =
            [&req](beast::string_view what)
            {
                http::response<http::string_body> res{http::status::internal_server_error, req.version()};
                res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
                res.set(http::field::content_type, "text/html");
                res.keep_alive(req.keep_alive());
                res.body() = "An error occurred: '" + std::string(what) + "'";
                res.prepare_payload();
                return res;
            };

    // Make sure we can handle the method
    if( req.method() != http::verb::get &&
        req.method() != http::verb::head)
        return send(bad_request("Unknown HTTP-method"));

    // Request path must be absolute and not contain "..".
    if( req.target().empty() ||
        req.target()[0] != '/' ||
        req.target().find("..") != beast::string_view::npos)
        return send(bad_request("Illegal request-target"));

    // Build the path to the requested file
    std::string path = path_cat(doc_root, req.target());
    if(req.target().back() == '/')
        path.append("index.html");

    // Attempt to open the file
    beast::error_code ec;
    http::file_body::value_type body;
    body.open(path.c_str(), beast::file_mode::scan, ec);

    // Handle the case where the file doesn't exist
    if(ec == beast::errc::no_such_file_or_directory)
        return send(not_found(req.target()));

    // Handle an unknown error
    if(ec)
        return send(server_error(ec.message()));

    // Cache the size since we need it after the move
    auto const size = body.size();

    // Respond to HEAD request
    if(req.method() == http::verb::head)
    {
        http::response<http::empty_body> res{http::status::ok, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, mime_type(path));
        res.content_length(size);
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }

    // Respond to GET request
    http::response<http::file_body> res{
            std::piecewise_construct,
            std::make_tuple(std::move(body)),
            std::make_tuple(http::status::ok, req.version())};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, mime_type(path));
    res.content_length(size);
    res.keep_alive(req.keep_alive());
    return send(std::move(res));
}

//------------------------------------------------------------------------------

// Report a failure
void
fail(beast::error_code ec, char const* what)
{
    // ssl::error::stream_truncated, also known as an SSL "short read",
    // indicates the peer closed the connection without performing the
    // required closing handshake (for example, Google does this to
    // improve performance). Generally this can be a security issue,
    // but if your communication protocol is self-terminated (as
    // it is with both HTTP and WebSocket) then you may simply
    // ignore the lack of close_notify.
    //
    // https://github.com/boostorg/beast/issues/38
    //
    // https://security.stackexchange.com/questions/91435/how-to-handle-a-malicious-ssl-tls-shutdown
    //
    // When a short read would cut off the end of an HTTP message,
    // Beast returns the error beast::http::error::partial_message.
    // Therefore, if we see a short read here, it has occurred
    // after the message has been completed, so it is safe to ignore it.

    if(ec == net::ssl::error::stream_truncated)
        return;

    std::cerr << what << ": " << ec.message() << "\n";
}

//------------------------------------------------------------------------------

// Handles an HTTP server connection.
// This uses the Curiously Recurring Template Pattern so that
// the same code works with both SSL streams and regular sockets.
template<class Derived>
class http_session
{
    // Access the derived class, this is part of
    // the Curiously Recurring Template Pattern idiom.
    Derived&
    derived()
    {
        return static_cast<Derived&>(*this);
    }

    // This queue is used for HTTP pipelining.
    class queue
    {
        enum
        {
            // Maximum number of responses we will queue
            limit = 8
        };

        // The type-erased, saved work item
        struct work
        {
            virtual ~work() = default;
            virtual void operator()() = 0;
        };

        http_session& self_;
        std::vector<std::unique_ptr<work>> items_;

    public:
        explicit
        queue(http_session& self)
                : self_(self)
        {
            static_assert(limit > 0, "queue limit must be positive");
            items_.reserve(limit);
        }

        // Returns `true` if we have reached the queue limit
        bool
        is_full() const
        {
            return items_.size() >= limit;
        }

        // Called when a message finishes sending
        // Returns `true` if the caller should initiate a read
        bool
        on_write()
        {
            BOOST_ASSERT(! items_.empty());
            auto const was_full = is_full();
            items_.erase(items_.begin());
            if(! items_.empty())
                (*items_.front())();
            return was_full;
        }

        // Called by the HTTP handler to send a response.
        template<bool isRequest, class Body, class Fields>
        void
        operator()(http::message<isRequest, Body, Fields>&& msg)
        {
            // This holds a work item
            struct work_impl : work
            {
                http_session& self_;
                http::message<isRequest, Body, Fields> msg_;

                work_impl(
                        http_session& self,
                        http::message<isRequest, Body, Fields>&& msg)
                        : self_(self)
                        , msg_(std::move(msg))
                {
                }

                void
                operator()()
                {
                    http::async_write(
                            self_.derived().stream(),
                            msg_,
                            beast::bind_front_handler(
                                    &http_session::on_write,
                                    self_.derived().shared_from_this(),
                                    msg_.need_eof()));
                }
            };

            // Allocate and store the work
            items_.push_back(
                    boost::make_unique<work_impl>(self_, std::move(msg)));

            // If there was no previous work, start this one
            if(items_.size() == 1)
                (*items_.front())();
        }
    };

    std::shared_ptr<std::string const> doc_root_;
    queue queue_;

    // The parser is stored in an optional container so we can
    // construct it from scratch it at the beginning of each new message.
    boost::optional<http::request_parser<http::string_body>> parser_;

protected:
    beast::flat_buffer buffer_;

public:
    // Construct the session
    http_session(
            beast::flat_buffer buffer,
            std::shared_ptr<std::string const> const& doc_root)
            : doc_root_(doc_root)
            , queue_(*this)
            , buffer_(std::move(buffer))
    {
    }

    void
    do_read()
    {
        // Construct a new parser for each message
        parser_.emplace();

        // Apply a reasonable limit to the allowed size
        // of the body in bytes to prevent abuse.
        parser_->body_limit(10000);

        // Set the timeout.
        beast::get_lowest_layer(
                derived().stream()).expires_after(std::chrono::seconds(30));

        // Read a request using the parser-oriented interface
        http::async_read(
                derived().stream(),
                buffer_,
                *parser_,
                beast::bind_front_handler(
                        &http_session::on_read,
                        derived().shared_from_this()));
    }

    void
    on_read(beast::error_code ec, std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // This means they closed the connection
        if(ec == http::error::end_of_stream)
            return derived().do_eof();

        if(ec)
            return fail(ec, "read");

        // Send the response
        handle_request(*doc_root_, parser_->release(), queue_);

        // If we aren't at the queue limit, try to pipeline another request
        if(! queue_.is_full())
            do_read();
    }

    void
    on_write(bool close, beast::error_code ec, std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        if(ec)
            return fail(ec, "write");

        if(close)
        {
            // This means we should close the connection, usually because
            // the response indicated the "Connection: close" semantic.
            return derived().do_eof();
        }

        // Inform the queue that a write completed
        if(queue_.on_write())
        {
            // Read another request
            do_read();
        }
    }
};

// Handles an SSL HTTP connection
class ssl_http_session
        : public http_session<ssl_http_session>
                , public std::enable_shared_from_this<ssl_http_session>
{
    beast::ssl_stream<beast::tcp_stream> stream_;

public:
    // Create the http_session
    ssl_http_session(
            beast::tcp_stream&& stream,
            ssl::context& ctx,
            beast::flat_buffer&& buffer,
            std::shared_ptr<std::string const> const& doc_root)
            : http_session<ssl_http_session>(
            std::move(buffer),
            doc_root)
            , stream_(std::move(stream), ctx)
    {
    }

    // Start the session
    void
    run()
    {
        // Set the timeout.
        beast::get_lowest_layer(stream_).expires_after(std::chrono::seconds(30));

        // Perform the SSL handshake
        // Note, this is the buffered version of the handshake.
        stream_.async_handshake(
                ssl::stream_base::server,
                buffer_.data(),
                beast::bind_front_handler(
                        &ssl_http_session::on_handshake,
                        shared_from_this()));
    }

    // Called by the base class
    beast::ssl_stream<beast::tcp_stream>&
    stream()
    {
        return stream_;
    }

    // Called by the base class
    beast::ssl_stream<beast::tcp_stream>
    release_stream()
    {
        return std::move(stream_);
    }

    // Called by the base class
    void
    do_eof()
    {
        // Set the timeout.
        beast::get_lowest_layer(stream_).expires_after(std::chrono::seconds(30));

        // Perform the SSL shutdown
        stream_.async_shutdown(
                beast::bind_front_handler(
                        &ssl_http_session::on_shutdown,
                        shared_from_this()));
    }

private:
    void
    on_handshake(
            beast::error_code ec,
            std::size_t bytes_used)
    {
        if(ec)
            return fail(ec, "handshake");

        // Consume the portion of the buffer used by the handshake
        buffer_.consume(bytes_used);

        do_read();
    }

    void
    on_shutdown(beast::error_code ec)
    {
        if(ec)
            return fail(ec, "shutdown");

        // At this point the connection is closed gracefully
    }
};

//------------------------------------------------------------------------------

// Detects SSL handshakes
class detect_session : public std::enable_shared_from_this<detect_session>
{
    beast::tcp_stream stream_;
    ssl::context& ctx_;
    std::shared_ptr<std::string const> doc_root_;
    beast::flat_buffer buffer_;

public:
    explicit
    detect_session(
            tcp::socket&& socket,
            ssl::context& ctx,
            std::shared_ptr<std::string const> const& doc_root)
            : stream_(std::move(socket))
            , ctx_(ctx)
            , doc_root_(doc_root)
    {
    }

    // Launch the detector
    void
    run()
    {
        // We need to be executing within a strand to perform async operations
        // on the I/O objects in this session. Although not strictly necessary
        // for single-threaded contexts, this example code is written to be
        // thread-safe by default.
        net::dispatch(
                stream_.get_executor(),
                beast::bind_front_handler(
                        &detect_session::on_run,
                        this->shared_from_this()));
    }

    void
    on_run()
    {
        // Set the timeout.
        stream_.expires_after(std::chrono::seconds(30));

        beast::async_detect_ssl(
                stream_,
                buffer_,
                beast::bind_front_handler(
                        &detect_session::on_detect,
                        this->shared_from_this()));
    }

    void
    on_detect(beast::error_code ec, bool result)
    {
        if(ec)
            return fail(ec, "detect");

        if(result)
        {
            // Launch SSL session
            std::make_shared<ssl_http_session>(
                    std::move(stream_),
                    ctx_,
                    std::move(buffer_),
                    doc_root_)->run();
            return;
        }
    }
};

// Accepts incoming connections and launches the sessions
class listener : public std::enable_shared_from_this<listener>
{
    net::io_context& ioc_;
    ssl::context& ctx_;
    tcp::acceptor acceptor_;
    std::shared_ptr<std::string const> doc_root_;

public:
    listener(
            net::io_context& ioc,
            ssl::context& ctx,
            tcp::endpoint endpoint,
            std::shared_ptr<std::string const> const& doc_root)
            : ioc_(ioc)
            , ctx_(ctx)
            , acceptor_(net::make_strand(ioc))
            , doc_root_(doc_root)
    {
        beast::error_code ec;

        // Open the acceptor
        acceptor_.open(endpoint.protocol(), ec);
        if(ec)
        {
            fail(ec, "open");
            return;
        }

        // Allow address reuse
        acceptor_.set_option(net::socket_base::reuse_address(true), ec);
        if(ec)
        {
            fail(ec, "set_option");
            return;
        }

        // Bind to the server address
        acceptor_.bind(endpoint, ec);
        if(ec)
        {
            fail(ec, "bind");
            return;
        }

        // Start listening for connections
        acceptor_.listen(
                net::socket_base::max_listen_connections, ec);
        if(ec)
        {
            fail(ec, "listen");
            return;
        }
    }

    // Start accepting incoming connections
    void
    run()
    {
        do_accept();
    }

private:
    void
    do_accept()
    {
        // The new connection gets its own strand
        acceptor_.async_accept(
                net::make_strand(ioc_),
                beast::bind_front_handler(
                        &listener::on_accept,
                        shared_from_this()));
    }

    void
    on_accept(beast::error_code ec, tcp::socket socket)
    {
        if(ec)
        {
            fail(ec, "accept");
        }
        else
        {
            // Create the detector http_session and run it
            std::make_shared<detect_session>(
                    std::move(socket),
                    ctx_,
                    doc_root_)->run();
        }

        // Accept another connection
        do_accept();
    }
};

//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    // Check command line arguments.
    /*
    if (argc != 5)
    {
        std::cerr <<
                  "Usage: advanced-server-flex <address> <port> <doc_root> <threads>\n" <<
                  "Example:\n" <<
                  "    advanced-server-flex 0.0.0.0 8080 . 1\n";
        return EXIT_FAILURE;
    }
     */
    auto const address = net::ip::make_address("127.0.0.1");
    auto const port = static_cast<unsigned short>(8080);
    auto const doc_root = std::make_shared<std::string>("/home/valera/DIPLOM/project/3D-reconstruction/html/");
    auto const threads = std::max<int>(1, 3);

    // The io_context is required for all I/O
    net::io_context ioc{threads};

    // The SSL context is required, and holds certificates
    ssl::context ctx{ssl::context::tlsv12};

    // This holds the self-signed certificate used by the server
    load_server_certificate(ctx);

    // Create and launch a listening port
    std::make_shared<listener>(
            ioc,
            ctx,
            tcp::endpoint{address, port},
            doc_root)->run();

    // Capture SIGINT and SIGTERM to perform a clean shutdown
    net::signal_set signals(ioc, SIGINT, SIGTERM);
    signals.async_wait(
            [&](beast::error_code const&, int)
            {
                // Stop the `io_context`. This will cause `run()`
                // to return immediately, eventually destroying the
                // `io_context` and all of the sockets in it.
                ioc.stop();
            });

    // Run the I/O service on the requested number of threads
    std::vector<std::thread> v;
    v.reserve(threads - 1);
    for(auto i = threads - 1; i > 0; --i)
        v.emplace_back(
                [&ioc]
                {
                    ioc.run();
                });
    ioc.run();

    // (If we get here, it means we got a SIGINT or SIGTERM)

    // Block until all the threads exit
    for(auto& t : v)
        t.join();

    boost::source_location a;

    return EXIT_SUCCESS;
}