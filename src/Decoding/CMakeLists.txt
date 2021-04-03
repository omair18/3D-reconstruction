set(DECODING_LIB_HEADERS
        IImageDecoder.h
        ImageDecoderFactory.h
        OpenCVImageDecoder.h
        NvJPEGImageDecoder.h
        NvJPEGHardwareImageDecoder.h
        NvJPEG2kImageDecoder.h)

set(DECODING_LIB_CPPS
        ImageDecoderFactory.cpp
        OpenCVImageDecoder.cpp
        NvJPEGImageDecoder.cpp
        NvJPEGHardwareImageDecoder.cpp
        NvJPEG2kImageDecoder.cpp)

source_group("Header Files" FILES ${DECODING_LIB_HEADERS})
source_group("Source Files" FILES ${DECODING_LIB_CPPS})

add_library(decoding SHARED ${DECODING_LIB_HEADERS} ${DECODING_LIB_CPPS})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(decoding
        PUBLIC

        PRIVATE
        log
        ${NVJPEG_LIBRARY}
        ${NVJPEG2K_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        opencv_core
        opencv_imgcodecs
        opencv_highgui
        datastructures)

target_include_directories(decoding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
