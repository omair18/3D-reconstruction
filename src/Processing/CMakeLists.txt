set(PROCESSING_LIB_HEADERS

        )

set(PROCESSING_LIB_CPPS

        )

source_group("Header Files" FILES ${PROCESSING_LIB_HEADERS})
source_group("Source Files" FILES ${PROCESSING_LIB_CPPS})

add_library(processing SHARED ${PROCESSING_LIB_HEADERS} ${PROCESSING_LIB_CPPS})

target_link_libraries(processing

    )

target_include_directories(processing PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})