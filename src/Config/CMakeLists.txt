set(CONFIG_LIB_HEADERS
        JsonConfig.h
        JsonConfigManager.h)

set(CONFIG_LIB_CPPS
        JsonConfig.cpp
        JsonConfigManager.cpp)

source_group("Header Files" FILES ${CONFIG_LIB_HEADERS})
source_group("Source Files" FILES ${CONFIG_LIB_CPPS})

add_library(config SHARED ${CONFIG_LIB_HEADERS} ${CONFIG_LIB_CPPS})

target_link_libraries(config
        PUBLIC
        stdc++fs
        PRIVATE
        log
        Boost::json)

target_include_directories(config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
