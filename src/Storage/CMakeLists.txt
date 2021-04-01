set(STORAGE_LIB_HEADERS
        )

set(STORAGE_LIB_CPPS
        )

source_group("Header Files" FILES ${STORAGE_LIB_HEADERS})
source_group("Source Files" FILES ${STORAGE_LIB_CPPS})

add_library(storage SHARED ${STORAGE_LIB_HEADERS} ${STORAGE_LIB_CPPS})

target_link_libraries(storage
        PUBLIC

        PRIVATE
        log
        config)

target_include_directories(storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
