set(PROCESSING_LIB_HEADERS
        IProcessor.h
        GpuProcessor.h
        ProcessorManager.h
        CpuProcessor.h
        Thread.h)

set(PROCESSING_LIB_CPPS
        IProcessor.cpp
        GpuProcessor.cpp
        ProcessorManager.cpp
        CpuProcessor.cpp
        Thread.cpp EndlessThread.cpp EndlessThread.h)

source_group("Header Files" FILES ${PROCESSING_LIB_HEADERS})
source_group("Source Files" FILES ${PROCESSING_LIB_CPPS})

add_library(processing SHARED ${PROCESSING_LIB_HEADERS} ${PROCESSING_LIB_CPPS})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(processing
        PUBLIC

        PRIVATE
        datastructures
        log
        algorithms
        config
        ${CUDA_CUDART_LIBRARY})

target_include_directories(processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})