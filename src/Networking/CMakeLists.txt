set(NETWORKING_LIB_HEADERS
        IWebServer.h
        HttpsServer.h
        HttpServer.h
        KafkaConsumer.h
        KafkaProducer.h)

set(NETWORKING_LIB_CPPS
        IWebServer.cpp
        HttpsServer.cpp
        HttpServer.cpp
        KafkaConsumer.cpp
        KafkaProducer.cpp)

source_group("Header Files" FILES ${NETWORKING_LIB_HEADERS})
source_group("Source Files" FILES ${NETWORKING_LIB_CPPS})

add_library(networking SHARED ${NETWORKING_LIB_HEADERS} ${NETWORKING_LIB_CPPS})

target_link_libraries(networking
        PUBLIC

        PRIVATE)

target_include_directories(networking PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})