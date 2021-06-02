set(STORAGE_LIB_HEADERS
#        AmazonBucketStorage.h
        FilesystemStorage.h
        IO/PLYModel.h)

set(STORAGE_LIB_CPPS
#        AmazonBucketStorage.cpp
        FilesystemStorage.cpp
        IO/PLYModel.cpp)

source_group("Header Files" FILES ${STORAGE_LIB_HEADERS})
source_group("Source Files" FILES ${STORAGE_LIB_CPPS})

add_library(storage SHARED ${STORAGE_LIB_HEADERS} ${STORAGE_LIB_CPPS})

target_link_libraries(storage
        PUBLIC

        PRIVATE
        networking
        log
        config)

target_include_directories(storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
