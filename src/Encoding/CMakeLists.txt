set(ENCODING_LIB_HEADERS
        IImageEncoder.h
        ImageEncoderFactory.h
        OpenCVImageEncoder.h
        NvJPEGImageEncoder.h
        NvJPEGHardwareImageEncoder.h
        NvJPEG2kImageEncoder.h)

set(ENCODING_LIB_CPPS
        ImageEncoderFactory.cpp
        OpenCVImageEncoder.cpp
        NvJPEGImageEncoder.cpp
        NvJPEGHardwareImageEncoder.cpp
        NvJPEG2kImageEncoder.cpp)

source_group("Header Files" FILES ${ENCODING_LIB_HEADERS})
source_group("Source Files" FILES ${ENCODING_LIB_CPPS})

add_library(encoding SHARED ${ENCODING_LIB_HEADERS} ${ENCODING_LIB_CPPS})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(encoding
        PUBLIC

        PRIVATE
        log
        )

target_include_directories(encoding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
