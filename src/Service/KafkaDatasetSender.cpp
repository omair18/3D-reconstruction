#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <librdkafka/rdkafkacpp.h>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/json/value.hpp>
#include <boost/json/object.hpp>
#include <boost/json/serialize.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>

int main(int argc, char** argv)
{
    std::filesystem::path inputPath;
    int frameStep;
    std::string kafkaBroker;
    std::string topic;
    int cameraId;
    int jpegQuality;
    float focalLength;
    float sensorSize;
    int timeout;

    boost::program_options::variables_map paramsMap;
    boost::program_options::options_description options("Options");
    options.add_options()
    ("help,h", "Produces help message.")
    ("input,i", boost::program_options::value<std::filesystem::path>(&inputPath)->required(), "Path to a folder with images or to a video.")
    ("frame-step,f", boost::program_options::value<int>(&frameStep)->required(), "Frame step.")
    ("topic,t", boost::program_options::value<std::string>(&topic)->required(), "Topic of Kafka broker.")
    ("broker,b", boost::program_options::value<std::string>(&kafkaBroker)->required(), "URL of Kafka broker.")
    ("camera-id,c", boost::program_options::value<int>(&cameraId)->required(), "Camera id.")
    ("jpeg-quality,j", boost::program_options::value<int>(&jpegQuality)->default_value(40), "JPEG quality.")
    ("focal-length,l", boost::program_options::value<float>(&focalLength)->default_value(40.0), "Focal length of the camera in millimeters.")
    ("sensor-size,s", boost::program_options::value<float>(&sensorSize)->default_value(33.3), "Camera sensor size in millimeters.")
    ("timeout,o", boost::program_options::value<int>(&timeout)->default_value(10000), "Kafka producer flush timeout in milliseconds");

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, options), paramsMap);

    if (paramsMap.count("help") || argc < 2)
    {
        std::clog << options << std::endl;
        return EXIT_SUCCESS;
    }

    boost::program_options::notify(paramsMap);

    if(!std::filesystem::exists(inputPath))
    {
        std::clog << "Invalid input path!" << std::endl;
        return EXIT_FAILURE;
    }

    if(frameStep <= 0)
    {
        std::clog << "Frame step must be positive!" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<cv::Mat> frames;
    auto datasetUUID = boost::lexical_cast<std::string>(boost::uuids::uuid(boost::uuids::random_generator()()));

    std::cout << "Reading input ..." << std::endl;

    if(std::filesystem::is_block_file(inputPath) ||
       std::filesystem::is_character_file(inputPath) ||
       std::filesystem::is_regular_file(inputPath))
    {
        auto capture = cv::VideoCapture(inputPath.string());
        if(!capture.isOpened())
        {
            std::clog << "Failed to read input file." << std::endl;
            return EXIT_FAILURE;
        }
        auto framesTotal = capture.get(cv::CAP_PROP_FRAME_COUNT);
        for(int i = 0; i < framesTotal; i += frameStep)
        {
            capture.set(cv::CAP_PROP_POS_FRAMES, i);
            cv::Mat image;
            capture >> image;
            if(!image.empty())
            {
                frames.push_back(image);
            }
            else
            {
                std::clog << "Failed to read frame with id " << i << std::endl;
            }
        }

    }
    else if (std::filesystem::is_directory(inputPath))
    {
        std::vector<std::filesystem::path> files;
        for(auto& file : std::filesystem::directory_iterator(inputPath))
        {
            if(std::filesystem::is_regular_file(file) ||
               std::filesystem::is_block_file(file)   ||
               std::filesystem::is_character_file(file))
            {
                files.push_back(file);
            }
        }

        auto filesCount = files.size();

        for(int i = 0; i < filesCount; i += frameStep)
        {
            cv::Mat image = cv::imread(files[i].string());
            if(!image.empty())
            {
                frames.push_back(image);
            }
            else
            {
                std::clog << "Failed to read image " << files[i] << std::endl;
            }
        }
    }
    else
    {
        std::clog << "Input path must be a path to either a video or a folder with images!" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Encoding images ..." << std::endl;

    auto imagesTotal = frames.size();
    std::vector<std::vector<uchar>> encodedImages;

    for(int i = 0; i < imagesTotal; ++i)
    {
        std::vector<uchar> buffer;
        if(!cv::imencode(".jpg", frames[i], buffer, {static_cast<int>(cv::IMWRITE_JPEG_QUALITY), jpegQuality}))
        {
            std::clog << "Failed to encode image with id = " << i << std::endl;
        }
        else
        {
            encodedImages.push_back(std::move(buffer));
        }
    }

    frames.clear();
    std::cout << "Sending images ..." << std::endl;

    std::string kafkaErrorString;

    RdKafka::Conf* topicConfig = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_TOPIC);
    RdKafka::Conf* globalConfig = RdKafka::Conf::create(RdKafka::Conf::ConfType::CONF_GLOBAL);

    globalConfig->set("default_topic_conf", topicConfig, kafkaErrorString);
    if(!kafkaErrorString.empty())
    {
        std::clog << "Kafka error: " << kafkaErrorString << std::endl;
        return EXIT_FAILURE;
    }
    delete topicConfig;

    globalConfig->set("metadata.broker.list", kafkaBroker, kafkaErrorString);
    if(!kafkaErrorString.empty())
    {
        std::clog << "Kafka error: " << kafkaErrorString << std::endl;
        return EXIT_FAILURE;
    }

    RdKafka::Producer* producer = RdKafka::Producer::create(globalConfig, kafkaErrorString);
    if(!kafkaErrorString.empty())
    {
        std::clog << "Kafka error: " << kafkaErrorString << std::endl;
        return EXIT_FAILURE;
    }

    delete globalConfig;

    auto encodedImagesTotal = encodedImages.size();
    for(int i = 0; i < encodedImagesTotal; ++i)
    {
        boost::json::value jsonKey;
        jsonKey.emplace_object();
        jsonKey.as_object().emplace("cameraID", cameraId);
        jsonKey.as_object().emplace("timestamp", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
        jsonKey.as_object().emplace("UUID", datasetUUID);
        jsonKey.as_object().emplace("frameID", i);
        jsonKey.as_object().emplace("framesTotal", encodedImagesTotal);
        jsonKey.as_object().emplace("focalLength", focalLength);
        jsonKey.as_object().emplace("sensorSize", sensorSize);
        auto jsonKeyString = boost::json::serialize(jsonKey);
        std::cout << jsonKeyString << std::endl;

        producer->produce(topic,
                          RdKafka::Topic::PARTITION_UA,
                          RdKafka::Producer::RK_MSG_COPY,
                          encodedImages[i].data(),
                          encodedImages[i].size(),
                          jsonKeyString.data(),
                          jsonKeyString.size(),
                          0,
                          nullptr,
                          nullptr);
        if(producer->flush(timeout) == RdKafka::ERR__TIMED_OUT)
        {
            std::clog << "Producing error.";
        }
    }

    delete producer;

    return EXIT_SUCCESS;
}