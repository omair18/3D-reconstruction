set(PROCESSING_LIB_HEADERS
        KafkaDatasetCollector.h)

set(PROCESSING_LIB_CPPS
        KafkaDatasetCollector.cpp)

source_group("Header Files" FILES ${PROCESSING_LIB_HEADERS})
source_group("Source Files" FILES ${PROCESSING_LIB_CPPS})

add_library(processing SHARED ${PROCESSING_LIB_HEADERS} ${PROCESSING_LIB_CPPS})

target_link_libraries(processing
        PUBLIC

        PRIVATE
        datastructures)

target_include_directories(processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})