set(UTILS_LIB_HEADERS
        FastPimpl.h
        StackTraceDumper.h
        UUIDGenerator.h
        PathUtils.h)

set(UTILS_LIB_CPPS
        StackTraceDumper.cpp
        UUIDGenerator.cpp
        PathUtils.cpp)

source_group("Header Files" FILES ${UTILS_LIB_HEADERS})
source_group("Source Files" FILES ${UTILS_LIB_CPPS})

add_library(utils SHARED ${UTILS_LIB_HEADERS} ${UTILS_LIB_CPPS})

target_link_libraries(utils
    PUBLIC

    PRIVATE
    Boost::stacktrace_addr2line
    log)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
