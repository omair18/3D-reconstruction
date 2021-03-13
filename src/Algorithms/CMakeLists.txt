set(ALGORITHMS_LIB_HEADERS
        IAlgorithm.h
        IAlgorithmFactory.h
        AlgorithmFactory.h
        KafkaIO/KafkaConsumptionAlgorithm.h
        KafkaIO/KafkaProducingAlgorithm.h
        Decoding/ImageDecodingAlgorithm.h
        Photogrammetry/DatasetCollectingAlgorithm.h)

set(ALGORITHMS_LIB_CPPS
        AlgorithmFactory.cpp
        KafkaIO/KafkaConsumptionAlgorithm.cpp
        KafkaIO/KafkaProducingAlgorithm.cpp
        Decoding/ImageDecodingAlgorithm.cpp
        Photogrammetry/DatasetCollectingAlgorithm.cpp)

source_group("Header Files" FILES ${ALGORITHMS_LIB_HEADERS})
source_group("Source Files" FILES ${ALGORITHMS_LIB_CPPS})

add_library(algorithms SHARED ${ALGORITHMS_LIB_HEADERS} ${ALGORITHMS_LIB_CPPS})

target_link_libraries(algorithms
        PUBLIC

        PRIVATE
        encoding
        decoding
        networking
        datastructures
        log
        )

target_include_directories(algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
