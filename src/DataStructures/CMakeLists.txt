set(DATA_STRUCTURES_LIB_HEADERS
        ModelDataset.h
        CUDAImage.h
        CUDAImageDescriptor.h
        ThreadSafeQueue.h)

set(DATA_STRUCTURES_LIB_CPPS
        ModelDataset.cpp
        CUDAImage.cpp
        CUDAImageDescriptor.cpp)

source_group("Header Files" FILES ${DATA_STRUCTURES_LIB_HEADERS})
source_group("Source Files" FILES ${DATA_STRUCTURES_LIB_CPPS})


add_library(datastructures SHARED ${DATA_STRUCTURES_LIB_HEADERS} ${DATA_STRUCTURES_LIB_CPPS})


target_link_libraries(datastructures
        PUBLIC

        PRIVATE

        )


target_include_directories(datastructures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
