set(GPU_LIB_HEADERS
    GpuManager.h
        )

set(GPU_LIB_CPPS
    GpuManager.cu
        )

source_group("Header Files" FILES ${CONFIG_LIB_HEADERS})
source_group("Source Files" FILES ${CONFIG_LIB_CPPS})

cuda_add_library(gpu SHARED ${GPU_LIB_HEADERS} ${GPU_LIB_CPPS})

target_link_libraries(gpu
    )
