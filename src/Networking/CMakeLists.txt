set(NETWORKING_LIB_HEADERS
    HttpServer.h
        )

set(NETWORKING_LIB_CPPS
    HttpServer.cpp
        )

source_group("Header Files" FILES ${NETWORKING_LIB_HEADERS})
source_group("Source Files" FILES ${NETWORKING_LIB_CPPS})

add_library(networking SHARED ${NETWORKING_LIB_HEADERS} ${NETWORKING_LIB_CPPS})

target_link_libraries(networking

    )
