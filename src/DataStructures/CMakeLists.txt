set(DATA_STRUCTURES_LIB_HEADERS
        ModelDataset.h
        CUDAImage.h
        CUDAImageDescriptor.h
        ProcessingQueue.h
        ProcessingQueueManager.h
        ProcessingData.h)

set(DATA_STRUCTURES_LIB_CPPS
        ModelDataset.cpp
        CUDAImage.cpp
        CUDAImageDescriptor.cpp
        ProcessingQueueManager.cpp
        ProcessingData.cpp)

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
        ${CUDA_CUDART_LIBRARY})


target_include_directories(datastructures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
