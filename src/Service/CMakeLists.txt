set(3D_RECONSTRUCTION_EXE_HEADERS
        ServiceSDK.h
        )

set(3D_RECONSTRUCTION_EXE_CPPS
        ServiceSDK.cpp
        Service.cpp
        )

source_group("Header Files" FILES ${3D_RECONSTRUCTION_EXE_HEADERS})
source_group("Source Files" FILES ${3D_RECONSTRUCTION_EXE_CPPS})

add_executable(Service ${3D_RECONSTRUCTION_EXE_HEADERS} ${3D_RECONSTRUCTION_EXE_CPPS})

target_link_libraries(Service
        PUBLIC

        PRIVATE
        networking
        stdc++fs
        utils
        config
        log
        gpu)

add_executable(KafkaDatasetSender KafkaDatasetSender.cpp)

target_link_libraries(KafkaDatasetSender Boost::program_options)

