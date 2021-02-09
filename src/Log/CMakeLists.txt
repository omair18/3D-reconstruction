set(LOG_LIB_HEADERS
        Logger.h
        )

set(LOG_LIB_CPPS
        Logger.cpp
        )

source_group("Header Files" FILES ${LOG_LIB_HEADERS})
source_group("Source Files" FILES ${LOG_LIB_CPPS})

add_library(log SHARED ${LOG_LIB_HEADERS} ${LOG_LIB_CPPS})

target_link_libraries(log PUBLIC Boost::log)

target_include_directories(log PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
