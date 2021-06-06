set(DATA_STRUCTURES_LIB_HEADERS
        ModelDataset.h
        CUDAImage.h
        ImageDescriptor.h
        ProcessingQueue.h
        ProcessingQueueManager.h
        ProcessingData.h
        # TODO: Implement library-independent key point class
        #KeyPoint.h
        ReconstructionParams.h
        # TODO: Remove OpenMVG interface
        RegionsProvider.h
        FundamentalMatrixRobustModelEstimator.h
        # TODO: Remove OpenMVS interface
        Scene.h
        DepthMapsData.h
        DenseDepthMapData.h
        DepthEstimator.h)

set(DATA_STRUCTURES_LIB_CPPS
        ModelDataset.cpp
        CUDAImage.cpp
        ImageDescriptor.cpp
        ProcessingQueueManager.cpp
        ProcessingData.cpp
        # TODO: Implement library-independent key point class
        #KeyPoint.cpp
        ReconstructionParams.cpp
        # TODO: Remove OpenMVG interface
        RegionsProvider.cpp
        FundamentalMatrixRobustModelEstimator.cpp
        # TODO: Remove OpenMVS interface
        Scene.cpp
        DepthMapsData.cpp
        DenseDepthMapData.cpp
        DepthEstimator.cpp)

source_group("Header Files" FILES ${DATA_STRUCTURES_LIB_HEADERS})
source_group("Source Files" FILES ${DATA_STRUCTURES_LIB_CPPS})


add_library(datastructures SHARED ${DATA_STRUCTURES_LIB_HEADERS} ${DATA_STRUCTURES_LIB_CPPS})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(datastructures
        PUBLIC

        PRIVATE
        log
        opencv_core
        config
        networking
        ${CUDA_CUDART_LIBRARY}
        OpenMVG::openMVG_sfm
        Common
        Math
        IO
        MVS)


target_include_directories(datastructures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
