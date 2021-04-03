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
        Photogrammetry/MeshReconstructionAlgorithm.h
        Photogrammetry/MeshRefinementAlgorithm.h
        Photogrammetry/MeshTexturingAlgorithm.h
        Photogrammetry/PointCloudDensificationAlgorithm.h
        ImageProcessing/CUDAResizeAlgorithm.h
        ImageProcessing/CUDAConvolutionAlgorithm.h
        ImageProcessing/CUDAImageBinarizationAlgorithm.h)

set(ALGORITHMS_LIB_CPPS
        AlgorithmFactory.cpp
        KafkaIO/KafkaConsumptionAlgorithm.cpp
        KafkaIO/KafkaProducingAlgorithm.cpp
        Decoding/CUDAImageDecodingAlgorithm.cpp
        Photogrammetry/DatasetCollectingAlgorithm.cpp
        Photogrammetry/AKAZEKeypointDetectionAlgorithm.cpp
        Photogrammetry/MeshReconstructionAlgorithm.cpp
        Photogrammetry/MeshRefinementAlgorithm.cpp
        Photogrammetry/MeshTexturingAlgorithm.cpp
        Photogrammetry/PointCloudDensificationAlgorithm.cpp
        ImageProcessing/CUDAResizeAlgorithm.cpp
        ImageProcessing/CUDAConvolutionAlgorithm.cpp
        ImageProcessing/CUDAImageBinarizationAlgorithm.cpp)

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
        opencv_core)

target_include_directories(algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
