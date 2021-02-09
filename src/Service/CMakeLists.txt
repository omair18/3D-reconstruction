set(3D_RECONSTRUCTION_EXE_HEADERS
        )

set(3D_RECONSTRUCTION_EXE_CPPS
        Service.cu
        )

source_group("Header Files" FILES ${3D_RECONSTRUCTION_EXE_HEADERS})
source_group("Source Files" FILES ${3D_RECONSTRUCTION_EXE_CPPS})

cuda_add_executable(Service ${3D_RECONSTRUCTION_EXE_HEADERS} ${3D_RECONSTRUCTION_EXE_CPPS})

target_link_libraries(Service
    utils
    log
    config
    )

