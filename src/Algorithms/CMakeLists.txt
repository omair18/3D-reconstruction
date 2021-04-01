set(ALGORITHMS_LIB_HEADERS
        IAlgorithm.h
        ICPUAlgorithm.h
        IGPUAlgorithm.h
        IAlgorithmFactory.h
        AlgorithmFactory.h
        KafkaIO/KafkaConsumptionAlgorithm.h
        KafkaIO/KafkaProducingAlgorithm.h
        Decoding/CUDAImageDecodingAlgorithm.h
        Photogrammetry/DatasetCollectingAlgorithm.h
        Photogrammetry/AKAZEKeypointDetectionAlgorithm.h
        ImageProcessing/CUDAResizeAlgorithm.h
        ImageProcessing/CUDAConvolutionAlgorithm.h)

set(ALGORITHMS_LIB_CPPS
        AlgorithmFactory.cpp
        KafkaIO/KafkaConsumptionAlgorithm.cpp
        KafkaIO/KafkaProducingAlgorithm.cpp
        Decoding/CUDAImageDecodingAlgorithm.cpp
        Photogrammetry/DatasetCollectingAlgorithm.cpp
        Photogrammetry/AKAZEKeypointDetectionAlgorithm.cpp
        ImageProcessing/CUDAResizeAlgorithm.cpp
        ImageProcessing/CUDAConvolutionAlgorithm.cpp ../Storage/FilesystemStorage.cpp ../Storage/FilesystemStorage.h ../Storage/AmazonBucketStorage.cpp ../Storage/AmazonBucketStorage.h ../Storage/IO/PLYModel.cpp ../Storage/IO/PLYModel.h Decoding/ImageDecodingAlgorithm.cpp Decoding/ImageDecodingAlgorithm.h)

source_group("Header Files" FILES ${ALGORITHMS_LIB_HEADERS})
source_group("Source Files" FILES ${ALGORITHMS_LIB_CPPS})

add_library(algorithms SHARED ${ALGORITHMS_LIB_HEADERS} ${ALGORITHMS_LIB_CPPS})

target_link_libraries(algorithms
        PUBLIC

        PRIVATE
        encoding
        decoding
        config
        networking
        datastructures
        log
        opencv_core
        )

target_include_directories(algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
