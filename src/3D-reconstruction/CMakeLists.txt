set(3D_RECONSTRUCTION_EXE_HEADERS
        )

set(3D_RECONSTRUCTION_EXE_CPPS
        main.cu
        )

source_group("Header Files" FILES ${3D_RECONSTRUCTION_EXE_HEADERS})
source_group("Source Files" FILES ${3D_RECONSTRUCTION_EXE_CPPS})

add_executable(3D-reconstruction ${3D_RECONSTRUCTION_EXE_HEADERS} ${3D_RECONSTRUCTION_EXE_CPPS})

target_link_libraries(3D-reconstruction
    ${CUDA_LIBRARIES})
