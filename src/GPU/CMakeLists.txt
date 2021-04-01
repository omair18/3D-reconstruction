set(GPU_LIB_HEADERS
        GpuManager.h
        GPU.h)

set(GPU_LIB_CPPS
        GpuManager.cpp)

source_group("Header Files" FILES ${GPU_LIB_HEADERS})
source_group("Source Files" FILES ${GPU_LIB_CPPS})

add_library(gpu SHARED ${GPU_LIB_HEADERS} ${GPU_LIB_CPPS})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(gpu
        PUBLIC

        PRIVATE
        ${CUDA_LIBRARIES}
        log
        config)

target_include_directories(gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
