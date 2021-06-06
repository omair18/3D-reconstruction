set(ALGORITHMS_LIB_HEADERS
        IAlgorithm.h
        ICPUAlgorithm.h
        IGPUAlgorithm.h
        IAlgorithmFactory.h
        AlgorithmFactory.h
        KafkaIO/KafkaConsumptionAlgorithm.h
        KafkaIO/KafkaProducingAlgorithm.h
        KafkaIO/KafkaMessageParsingAlgorithm.h
        Decoding/CUDAImageDecodingAlgorithm.h
        Decoding/CpuImageDecodingAlgorithm.h
        Photogrammetry/DatasetCollectingAlgorithm.h
        #Photogrammetry/CUDAAKAZEKeypointDetectionAlgorithm.h
        Photogrammetry/AKAZEKeypointDetectionAlgorithm.h
        #Photogrammetry/CUDAKeyPointMatchingAlgorithm.h
        Photogrammetry/KeyPointMatchingAlgorithm.h
        #Photogrammetry/CUDAKeyPointFilteringAlgorithm.h
        Photogrammetry/KeyPointFilteringAlgorithm.h
        #Photogrammetry/CUDABundleAdjustmentAlgorithm.h
        Photogrammetry/BundleAdjustmentAlgorithm.h
        #Photogrammetry/CUDAPointCloudDensificationAlgorithm.h
        Photogrammetry/PointCloudDensificationAlgorithm.h
        #Photogrammetry/CUDAMeshReconstructionAlgorithm.h
        Photogrammetry/MeshReconstructionAlgorithm.h
        #Photogrammetry/CUDAMeshRefinementAlgorithm.h
        #Photogrammetry/MeshRefinementAlgorithm.h
        #Photogrammetry/CUDAMeshTexturingAlgorithm.h
        Photogrammetry/MeshTexturingAlgorithm.h
# TODO: Implement image processing algorithms
#        ImageProcessing/CUDAImageResizeAlgorithm.h
#        ImageProcessing/ImageResizeAlgorithm.h
#        ImageProcessing/CUDAImageConvolutionAlgorithm.h
#        ImageProcessing/ImageConvolutionAlgorithm.h
#        ImageProcessing/CUDAImageBinarizationAlgorithm.h
#        ImageProcessing/ImageBinarizationAlgorithm.h
#        ImageProcessing/CUDAImageSeparableConvolutionAlgorithm.h
#        ImageProcessing/ImageSeparableConvolutionAlgorithm.h
        DataTransferring/DeviceToHostTransferringAlgorithm.h
        DataTransferring/HostToDeviceTransferringAlgorithm.h)

set(ALGORITHMS_LIB_CPPS
        AlgorithmFactory.cpp
        KafkaIO/KafkaConsumptionAlgorithm.cpp
        KafkaIO/KafkaProducingAlgorithm.cpp
        KafkaIO/KafkaMessageParsingAlgorithm.cpp
        Decoding/CUDAImageDecodingAlgorithm.cpp
        Decoding/CpuImageDecodingAlgorithm.cpp
        Photogrammetry/DatasetCollectingAlgorithm.cpp
        #Photogrammetry/CUDAAKAZEKeypointDetectionAlgorithm.cpp
        Photogrammetry/AKAZEKeypointDetectionAlgorithm.cpp
        #Photogrammetry/CUDAKeyPointMatchingAlgorithm.cpp
        Photogrammetry/KeyPointMatchingAlgorithm.cpp
        #Photogrammetry/CUDAKeyPointFilteringAlgorithm.cpp
        Photogrammetry/KeyPointFilteringAlgorithm.cpp
        #Photogrammetry/CUDABundleAdjustmentAlgorithm.cpp
        Photogrammetry/BundleAdjustmentAlgorithm.cpp
        #Photogrammetry/CUDAPointCloudDensificationAlgorithm.cpp
        Photogrammetry/PointCloudDensificationAlgorithm.cpp
        #Photogrammetry/CUDAMeshReconstructionAlgorithm.cpp
        Photogrammetry/MeshReconstructionAlgorithm.cpp
        #Photogrammetry/CUDAMeshRefinementAlgorithm.cpp
        #Photogrammetry/MeshRefinementAlgorithm.cpp
        #Photogrammetry/CUDAMeshTexturingAlgorithm.cpp
        Photogrammetry/MeshTexturingAlgorithm.cpp
# TODO: Implement image processing algorithms
#        ImageProcessing/CUDAImageResizeAlgorithm.cpp
#        ImageProcessing/ImageResizeAlgorithm.cpp
#        ImageProcessing/CUDAImageConvolutionAlgorithm.cpp
#        ImageProcessing/ImageConvolutionAlgorithm.cpp
#        ImageProcessing/CUDAImageBinarizationAlgorithm.cpp
#        ImageProcessing/ImageBinarizationAlgorithm.cpp
#        ImageProcessing/CUDAImageSeparableConvolutionAlgorithm.cpp
#        ImageProcessing/ImageSeparableConvolutionAlgorithm.cpp
        DataTransferring/DeviceToHostTransferringAlgorithm.cpp
        DataTransferring/HostToDeviceTransferringAlgorithm.cpp Photogrammetry/DatasetInitializationAlgorithm.cpp Photogrammetry/DatasetInitializationAlgorithm.h)

source_group("Header Files" FILES ${ALGORITHMS_LIB_HEADERS})
source_group("Source Files" FILES ${ALGORITHMS_LIB_CPPS})

include_directories(${CUDA_INCLUDE_DIRS})

add_library(algorithms SHARED ${ALGORITHMS_LIB_HEADERS} ${ALGORITHMS_LIB_CPPS})

target_link_libraries(algorithms
        PUBLIC

        PRIVATE
        encoding
        decoding
        config
        networking
        datastructures
        gpu
        utils
        log
        opencv_core
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppig_LIBRARY}
        ${CUDA_nppif_LIBRARY}
        ${CUDA_nppicc_LIBRARY}
        ${CUDA_nppist_LIBRARY}
        ${CUDA_nppidei_LIBRARY}
        ${CUDA_nppial_LIBRARY}
        OpenMVG::openMVG_sfm
        OpenMVG::openMVG_camera
        OpenMVG::openMVG_matching_image_collection
        opencv_features2d
        stdc++fs)

target_include_directories(algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
