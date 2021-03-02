set(NETWORKING_LIB_HEADERS
        WebServer.h
        WebServerFactory.h
        HttpsSession.h
        HttpSession.h
        KafkaConsumer.h
        KafkaProducer.h
        KafkaMessage.h
        EndpointListener.h)

set(NETWORKING_LIB_CPPS
        WebServer.cpp
        WebServerFactory.cpp
        HttpsSession.cpp
        HttpSession.cpp
        KafkaConsumer.cpp
        KafkaProducer.cpp
        KafkaMessage.cpp
        EndpointListener.cpp)

source_group("Header Files" FILES ${NETWORKING_LIB_HEADERS})
source_group("Source Files" FILES ${NETWORKING_LIB_CPPS})

add_library(networking SHARED ${NETWORKING_LIB_HEADERS} ${NETWORKING_LIB_CPPS})

target_link_libraries(networking
        PUBLIC

        PRIVATE
        log
        utils
        config
        pthread
        ssl
        crypto
        Boost::system
        RdKafka::rdkafka++
        stdc++fs)

target_include_directories(networking PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})